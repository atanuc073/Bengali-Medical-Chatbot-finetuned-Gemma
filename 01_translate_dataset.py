"""
Step 1: Translate ChatDoctor-HealthCareMagic-100k from English to Bengali
using AI4Bharat IndicTrans2.

Usage:
    python 01_translate_dataset.py                       # Full dataset
    python 01_translate_dataset.py --max_samples 10000   # First 10K rows
    python 01_translate_dataset.py --resume               # Resume from checkpoint
"""

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ─────────────────────────────── Config ───────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "ben_Beng"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bengali_medical_dataset.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "translation_checkpoint.json")

BATCH_SIZE = 8          # Reduce if OOM; increase if you have more VRAM
MAX_LENGTH = 512        # Max tokens per translation


def parse_args():
    parser = argparse.ArgumentParser(description="Translate medical dataset to Bengali")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N samples (default: all)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Translation batch size (default: 8)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    return parser.parse_args()


def load_translation_model():
    """Load IndicTrans2 model, tokenizer, and processor."""
    print(f"Loading IndicTrans2 model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    ip = IndicProcessor(inference=True)
    print("Model loaded successfully!")
    return model, tokenizer, ip


def translate_batch(sentences, model, tokenizer, ip):
    """Translate a batch of English sentences to Bengali."""
    if not sentences:
        return []

    # Pre-process
    batch = ip.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # Tokenize
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_attention_mask=True
    ).to(DEVICE)



    # Generate
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=MAX_LENGTH,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode & post-process
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
    return translations


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load dataset ──
    print("Loading ChatDoctor-HealthCareMagic-100k dataset ...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    total = len(dataset)
    print(f"Total samples to translate: {total}")

    # ── Resume logic ──
    start_idx = 0
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            ckpt = json.load(f)
            start_idx = ckpt.get("last_processed", 0)
        print(f"Resuming from index {start_idx}")

    # ── Load model ──
    model, tokenizer, ip = load_translation_model()

    # ── Translate in batches ──
    mode = "a" if args.resume and start_idx > 0 else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out_f:
        for i in range(start_idx, total, args.batch_size):
            batch_end = min(i + args.batch_size, total)
            batch_rows = dataset[i:batch_end]

            # Collect texts to translate
            instructions = batch_rows["instruction"]
            inputs_text = batch_rows["input"]
            outputs_text = batch_rows["output"]

            # Translate each field
            bn_instructions = translate_batch(instructions, model, tokenizer, ip)
            bn_inputs = translate_batch(inputs_text, model, tokenizer, ip)
            bn_outputs = translate_batch(outputs_text, model, tokenizer, ip)

            # Write translated rows
            for j in range(len(bn_instructions)):
                row = {
                    "instruction": bn_instructions[j],
                    "input": bn_inputs[j],
                    "output": bn_outputs[j],
                    "instruction_en": instructions[j],
                    "input_en": inputs_text[j],
                    "output_en": outputs_text[j],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Save checkpoint
            with open(CHECKPOINT_FILE, "w") as ckpt_f:
                json.dump({"last_processed": batch_end}, ckpt_f)

            progress = (batch_end / total) * 100
            print(f"[{progress:5.1f}%] Translated {batch_end}/{total} samples")

    print(f"\n✅ Translation complete! Saved to: {OUTPUT_FILE}")
    print(f"   Total translated: {total - start_idx} rows")


if __name__ == "__main__":
    main()
