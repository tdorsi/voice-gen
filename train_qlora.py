#!/usr/bin/env python3
"""
train_qlora.py — QLoRA fine-tuning for MOSS-TTS on single GPU (12 GB VRAM)
===========================================================================
Loads the 8B model in 4-bit (NF4) via bitsandbytes (~4 GB), then trains only
small LoRA adapter weights — fits on an RTX 5070 (11.94 GB) even with the
audio tokenizer resident in VRAM.

Usage (called by voice_gen.py stage 8):
    python train_qlora.py \\
        --model-path  D:/AI_Models/Voice/moss-tts/weights/MOSS-TTS-HF \\
        --codec-path  D:/AI_Models/Voice/moss-tts/weights/MOSS-Audio-Tokenizer-HF \\
        --train-jsonl D:/AI_Models/Voice/moss-tts/voices/output/Lori/train_with_codes.jsonl \\
        --output-dir  D:/AI_Models/Voice/moss-tts/voices/output/Lori/checkpoint \\
        [--epochs 3] [--lr 1e-5] [--lora-r 8] [--lora-alpha 16]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, get_scheduler

# ── repo on sys.path ──────────────────────────────────────────────────────────

MOSS_REPO = Path(r"D:\AI_Models\Voice\moss-tts\repo")
if str(MOSS_REPO) not in sys.path:
    sys.path.insert(0, str(MOSS_REPO))

from moss_tts_delay.finetuning.common import load_jsonl, normalize_audio_path_list
from moss_tts_delay.finetuning.dataset import MossTTSSFTDataset
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor

# ── LoRA target modules (same as community train_lora.py) ─────────────────────

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for MOSS-TTS")
    p.add_argument("--model-path",  required=True)
    p.add_argument("--codec-path",  required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora-r",      type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--max-seq-len", type=int,   default=2048)
    p.add_argument("--save-steps",  type=int,   default=0,
                   help="Save adapter checkpoint every N optimizer steps (0 = only at end)")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--channelwise-loss-weight", type=str, default="1,32")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return time.strftime("%H:%M:%S")


def _needs_audio_tokenizer(records: list) -> bool:
    for rec in records:
        ref = normalize_audio_path_list(rec.get("ref_audio"), "ref_audio")
        if rec.get("ref_audio_codes") is None and ref is not None:
            return True
    return False


def parse_channelwise_loss_weight(spec: str, n_heads: int) -> list:
    vals = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if len(vals) == n_heads:
        return vals
    if len(vals) == 2:
        text_w, audio_w = vals
        per_audio = audio_w / max(1, n_heads - 1)
        return [text_w] + [per_audio] * (n_heads - 1)
    raise ValueError(
        f"channelwise-loss-weight expects {n_heads} or 2 values, got {len(vals)}"
    )


# ── LoRA setup (matches community train_lora.py patches) ─────────────────────

def apply_lora(model: torch.nn.Module, args: argparse.Namespace):
    """Freeze base weights, apply LoRA adapter, patch PEFT quirks."""
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,  # MossTTSDelayModel has no prepare_inputs_for_generation; use base PeftModel
    )

    # Monkey-patch: PEFT calls get_input_embeddings() with no args during setup,
    # but MossTTSDelayModel.get_input_embeddings(input_ids) requires input_ids.
    original_get_input_embeddings = type(model).get_input_embeddings
    type(model).get_input_embeddings = lambda self, input_ids=None: (
        original_get_input_embeddings(self, input_ids)
        if input_ids is not None
        else self.language_model.get_input_embeddings()
    )

    model = get_peft_model(model, lora_config)

    # Monkey-patch: PEFT passes output_hidden_states in kwargs, which collides
    # with the explicit output_hidden_states=True call inside MossTTSDelayModel.
    _orig_forward = type(
        model.get_base_model() if hasattr(model, "get_base_model") else model
    ).forward

    def _patched_forward(self, *a, output_hidden_states=None, return_dict=None, **kw):
        return _orig_forward(self, *a, **kw)

    base_cls = type(
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    base_cls.forward = _patched_forward

    # Restrict trainable params to LoRA weights inside language_model layers.
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        in_scope = "language_model.layers." in name
        param.requires_grad = is_lora and in_scope

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{ts()}] Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / max(1, total):.3f}%)")

    if trainable == 0:
        raise RuntimeError(
            "No trainable LoRA parameters found — check target_modules."
        )
    return model


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{ts()}] Device: {device}")

    # ── load data ────────────────────────────────────────────────────────────
    records = load_jsonl(Path(args.train_jsonl))
    if not records:
        raise ValueError(f"No records in {args.train_jsonl}")
    print(f"[{ts()}] Loaded {len(records)} training records")

    need_codec = _needs_audio_tokenizer(records)
    print(f"[{ts()}] Audio tokenizer needed (ref encoding): {need_codec}")

    # ── build processor ──────────────────────────────────────────────────────
    if need_codec:
        processor = MossTTSDelayProcessor.from_pretrained(
            args.model_path,
            codec_path=args.codec_path,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    else:
        from transformers import AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        processor = MossTTSDelayProcessor(
            tokenizer=tokenizer,
            audio_tokenizer=None,
            model_config=config,
        )

    dataset = MossTTSSFTDataset(records=records, processor=processor)

    # Free codec VRAM before loading the big model
    if need_codec and getattr(processor, "audio_tokenizer", None) is not None:
        processor.audio_tokenizer = None
        torch.cuda.empty_cache()
        print(f"[{ts()}] Codec released from VRAM")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    # ── load model in 4-bit ──────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[{ts()}] Loading model in 4-bit NF4 from {args.model_path} ...")
    model = MossTTSDelayModel.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
        device_map={"": device},
    )
    print(f"[{ts()}] Model loaded. "
          f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = apply_lora(model, args)
    model.train()

    # ── channelwise loss weight ───────────────────────────────────────────────
    n_heads = model.get_base_model().config.n_vq + 1
    cw_weights = parse_channelwise_loss_weight(args.channelwise_loss_weight, n_heads)
    cw_tensor = torch.tensor(cw_weights, dtype=torch.float32, device=device)
    print(f"[{ts()}] channelwise_loss_weight ({n_heads} heads): {cw_weights[:4]}...")

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    micro_steps_per_epoch = len(dataloader)
    update_steps_per_epoch = math.ceil(micro_steps_per_epoch / args.grad_accum)
    max_train_steps = args.epochs * update_steps_per_epoch
    warmup_steps = math.ceil(max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    print(f"[{ts()}] Training: {args.epochs} epochs, "
          f"{micro_steps_per_epoch} micro-batches/epoch, "
          f"{update_steps_per_epoch} optimizer steps/epoch, "
          f"{max_train_steps} total steps, warmup={warmup_steps}")

    # ── training loop ─────────────────────────────────────────────────────────
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    channelwise_loss_weight=cw_tensor,
                )
                loss = outputs.loss / args.grad_accum

            loss.backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            elapsed = time.perf_counter() - t0
            avg_loss = accum_loss  # already divided by grad_accum
            accum_loss = 0.0

            print(
                f"[{ts()}] epoch={epoch+1}/{args.epochs} "
                f"step={global_step}/{max_train_steps} "
                f"loss={avg_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.0f}s "
                f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB"
            )
            sys.stdout.flush()

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt = output_dir / f"adapter-step-{global_step}"
                model.save_pretrained(str(ckpt))
                print(f"[{ts()}] Checkpoint -> {ckpt}")

        if global_step >= max_train_steps:
            break

    # ── save final adapter ────────────────────────────────────────────────────
    final_dir = output_dir / "adapter-final"
    model.save_pretrained(str(final_dir))
    # Write path file before any print so stage 9 finds it even if console encoding fails
    (output_dir / "adapter_path.txt").write_text(str(final_dir))
    print(f"[{ts()}] Training complete. Adapter -> {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
