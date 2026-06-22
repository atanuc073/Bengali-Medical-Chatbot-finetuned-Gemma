"""
Step 2: Fine-tune Gemma 1B (instruction-tuned) on Bengali medical Q&A
using Unsloth + QLoRA (4-bit) + SFTTrainer.

Unsloth provides 2x faster training and 60% less VRAM compared to
standard HuggingFace fine-tuning.

Usage:
    python 02_finetune_gemma.py
    python 02_finetune_gemma.py --epochs 3 --lr 2e-4
    python 02_finetune_gemma.py --dataset_path data/bengali_medical_dataset.jsonl
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────── Config ───────────────────────────────

MODEL_NAME = "unsloth/gemma-3-1b-it"       # Unsloth optimized Gemma 3 1B
DATASET_PATH = "data/bengali_medical_dataset.jsonl"
OUTPUT_DIR = "output/gemma-1b-bengali-medical"
MAX_SEQ_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma on Bengali medical data (Unsloth)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--save_gguf", action="store_true",
                        help="Also export model in GGUF format for Ollama/llama.cpp")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models")
    return parser.parse_args()


def load_bengali_dataset(path):
    """Load the translated Bengali medical JSONL dataset."""
    print(f"📂 Loading dataset from: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    print(f"   Loaded {len(dataset)} samples")
    return dataset


def format_to_conversations(example):
    """
    Convert instruction/input/output format to conversation format
    for Unsloth's chat template system.
    
    Each example becomes:
        [
            {"role": "user", "content": "<instruction>\n<input>"},
            {"role": "assistant", "content": "<output>"},
        ]
    """
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    # Combine instruction and input for the user turn
    if user_input.strip():
        user_message = f"{instruction}\n{user_input}"
    else:
        user_message = instruction

    example["conversations"] = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": output},
    ]
    return example


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── HuggingFace Login for Gated Model access ──
    token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Check Kaggle Secrets if not provided
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
                print("🔑 Loaded HF_TOKEN from Kaggle Secrets.")
        except Exception:
            pass

    if token:
        try:
            from huggingface_hub import login
            login(token=token)
            print("🔓 Logged in to Hugging Face successfully.")
        except Exception as e:
            print(f"⚠️ Failed to log in to Hugging Face: {e}")

    # ── Step 1: Load Model with Unsloth ──
    print(f"\n🚀 Loading model: {args.model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,          # QLoRA 4-bit quantization
        dtype=None,                 # Auto-detect (bf16 / fp16)
        token=token,                # Pass token explicitly to FastModel
    )

    # ── Step 2: Configure LoRA adapters ──
    print("🔧 Applying LoRA adapters ...")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",   # Unsloth's optimized GC
        random_state=42,
    )

    # ── Step 3: Setup chat template ──
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    # ── Step 4: Load & format dataset ──
    dataset = load_bengali_dataset(args.dataset_path)
    dataset = dataset.map(format_to_conversations, remove_columns=[
        col for col in dataset.column_names if col != "conversations"
    ])

    # Apply chat template to tokenize conversations
    def apply_template(examples):
        texts = []
        for convo in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True, remove_columns=["conversations"])

    # Shuffle and split: 95% train, 5% eval
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"   Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # ── Step 5: Training config ──
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        optim="adamw_8bit",
        max_grad_norm=0.3,
        seed=42,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",          # Change to "wandb" if using W&B
    )

    # ── Step 6: Create SFT Trainer ──
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # Show GPU stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"\n📊 GPU: {gpu_stats.name} | VRAM: {gpu_stats.total_mem / 1024**3:.1f} GB")
    print(f"   Reserved before training: {reserved_mem} GB")

    # ── Step 7: Train! ──
    print("\n🔥 Starting training ...")
    trainer_stats = trainer.train()

    # Post-training stats
    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"\n📊 Peak GPU memory used: {used_mem} GB")
    print(f"   Training time: {trainer_stats.metrics['train_runtime']:.0f}s")

    # ── Step 8: Save model ──
    print(f"\n💾 Saving LoRA adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optional: Save merged model in HuggingFace format (16-bit)
    merged_dir = os.path.join(args.output_dir, "merged_16bit")
    print(f"💾 Saving merged model (16-bit) to: {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # Optional: Export GGUF for Ollama / llama.cpp
    if args.save_gguf:
        gguf_dir = os.path.join(args.output_dir, "gguf")
        print(f"💾 Exporting GGUF to: {gguf_dir}")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=["q4_k_m", "q8_0"],
        )

    print("\n✅ Fine-tuning complete!")
    print(f"   LoRA adapter:  {args.output_dir}")
    print(f"   Merged model:  {merged_dir}")


if __name__ == "__main__":
    main()
