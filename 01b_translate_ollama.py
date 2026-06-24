"""
Step 1b: Translate ChatDoctor-HealthCareMagic-100k from English to Bengali
using a local Ollama Gemma3:4b model via instruction-based prompting.

Medical terms, disease names, drug names, and procedure names are kept in
English to preserve clinical accuracy.

Requirements:
    - Ollama installed and running locally (https://ollama.com)
    - Model pulled:  ollama pull gemma3:4b

Usage:
    python 01b_translate_ollama.py                        # Full dataset
    python 01b_translate_ollama.py --max_samples 1000     # First 1K rows
    python 01b_translate_ollama.py --resume                # Resume from checkpoint
    python 01b_translate_ollama.py --push_to_hub --hub_repo_id "Atanuc73/Bengali-Medical-Chatbot-Dataset"
"""

import os
import json
import time
import argparse
import requests
from datasets import load_dataset

# ─────────────────────────────── Config ───────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bengali_medical_dataset.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "translation_checkpoint.json")

SYSTEM_PROMPT = """You are an expert English-to-Bengali medical translator.

RULES:
1. Translate the given English text into natural, fluent Bengali.
2. **DO NOT translate** the following — keep them exactly in English:
   - Disease names (e.g. diabetes, BPPV, pneumonia, GERD, scabies)
   - Drug / medicine names (e.g. Omeprazole, Metformin, Amoxicillin)
   - Medical procedure names (e.g. MRI, ECG, X-ray, biopsy, endoscopy)
   - Medical abbreviations (e.g. BP, ICU, OPD, ENT, CT scan)
   - Anatomical terms when commonly used in English (e.g. cervical, lumbar)
3. Output ONLY the Bengali translation. No explanations, no preamble, no notes.
4. Preserve the original meaning and tone accurately.
5. If the input is empty or just whitespace, return an empty string."""

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate medical dataset to Bengali using Ollama Gemma3:4b"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to first N samples (default: all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--ollama_url", type=str, default=OLLAMA_BASE_URL,
        help="Ollama API base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help="Ollama model name (default: gemma3:4b)"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push the translated dataset to Hugging Face Hub when complete"
    )
    parser.add_argument(
        "--hub_repo_id", type=str, default=None,
        help="Hugging Face repo ID to push to (e.g. Atanuc73/Bengali-Medical-Chatbot-Dataset)"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)"
    )
    return parser.parse_args()


def check_ollama(base_url):
    """Verify Ollama is running and the model is available."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return models
    except requests.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        print("   Start it with: ollama serve")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Ollama health check failed: {e}")
        raise SystemExit(1)


def translate_text(text, base_url, model):
    """Translate a single English text to Bengali via Ollama chat API."""
    if not text or not text.strip():
        return ""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 2048,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            response = r.json()
            return response["message"]["content"].strip()
        except requests.Timeout:
            print(f"  ⏱️ Timeout on attempt {attempt}/{MAX_RETRIES}, retrying...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  ⚠️ Error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY)

    print(f"  ❌ Failed after {MAX_RETRIES} attempts. Returning original text.")
    return text


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Check Ollama ──
    print(f"Checking Ollama at {args.ollama_url} ...")
    available_models = check_ollama(args.ollama_url)
    print(f"Available models: {available_models}")

    # Verify model is pulled
    model_found = any(args.model in m for m in available_models)
    if not model_found:
        print(f"⚠️ Model '{args.model}' not found. Pulling it now...")
        os.system(f"ollama pull {args.model}")

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

    # ── Translate one-by-one ──
    mode = "a" if args.resume and start_idx > 0 else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out_f:
        for i in range(start_idx, total):
            sample = dataset[i]
            instruction_en = sample["instruction"]
            input_en = sample["input"]
            output_en = sample["output"]

            # Translate each field
            bn_instruction = translate_text(instruction_en, args.ollama_url, args.model)
            bn_input = translate_text(input_en, args.ollama_url, args.model)
            bn_output = translate_text(output_en, args.ollama_url, args.model)

            row = {
                "instruction": bn_instruction,
                "input": bn_input,
                "output": bn_output,
                "instruction_en": instruction_en,
                "input_en": input_en,
                "output_en": output_en,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            # Save checkpoint every sample (cheap for sequential processing)
            with open(CHECKPOINT_FILE, "w") as ckpt_f:
                json.dump({"last_processed": i + 1}, ckpt_f)

            progress = ((i + 1) / total) * 100
            if (i + 1) % 10 == 0 or i == start_idx:
                print(f"[{progress:5.1f}%] Translated {i + 1}/{total} samples")

    print(f"\n✅ Translation complete! Saved locally to: {OUTPUT_FILE}")
    print(f"   Total translated: {total - start_idx} rows")

    # ── Push to Hub ──
    if args.push_to_hub and args.hub_repo_id:
        print(f"\nUploading dataset to Hugging Face Hub ({args.hub_repo_id})...")
        try:
            token = args.hf_token or os.environ.get("HF_TOKEN")
            final_dataset = load_dataset("json", data_files=OUTPUT_FILE, split="train")
            final_dataset.push_to_hub(args.hub_repo_id, token=token)
            print(f"✅ Successfully pushed to https://huggingface.co/datasets/{args.hub_repo_id}")
        except Exception as e:
            print(f"⚠️ Failed to push to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
