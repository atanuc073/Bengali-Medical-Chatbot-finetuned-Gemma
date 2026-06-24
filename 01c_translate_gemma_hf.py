"""
Step 1c: Translate ChatDoctor-HealthCareMagic-100k from English to Bengali
using Google Gemma-3-4B-IT via HuggingFace Transformers on Kaggle (2x T4 GPUs).

Medical terms, disease names, drug names, and procedure names are kept in
English to preserve clinical accuracy.

Requirements:
    - transformers>=4.49.0
    - torch, accelerate, datasets, huggingface_hub
    - 2x T4 GPUs (Kaggle)
    - HF_TOKEN with access to google/gemma-3-4b-it

Usage (Kaggle notebook cell):
    !python 01c_translate_gemma_hf.py --max_samples 1000
    !python 01c_translate_gemma_hf.py --resume
    !python 01c_translate_gemma_hf.py --push_to_hub --hub_repo_id "Atanuc73/Bengali-Medical-Chatbot-Dataset"
"""

import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────── Config ───────────────────────────────

MODEL_ID = "google/gemma-3-4b-it"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bengali_medical_dataset.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "translation_checkpoint.json")

BATCH_SIZE = 4  # per GPU — kept small due to long generation lengths

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate medical dataset to Bengali using Gemma-3-4B-IT on Kaggle"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to first N samples (default: all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Translation batch size per GPU (default: 4)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated model access"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push the translated dataset to Hugging Face Hub when complete"
    )
    parser.add_argument(
        "--hub_repo_id", type=str, default=None,
        help="Hugging Face repo ID (e.g. Atanuc73/Bengali-Medical-Chatbot-Dataset)"
    )
    return parser.parse_args()


def get_hf_token(cli_token=None):
    """Resolve HF token from CLI arg, env var, or Kaggle Secrets."""
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
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
            print(f"⚠️ HF login failed: {e}")
    return token


def load_models(token):
    """Load Gemma-3-4B-IT on each available GPU."""
    print(f"Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    models = []

    if num_gpus > 1:
        print(f"Found {num_gpus} GPUs. Loading model on each GPU...")
        for i in range(num_gpus):
            print(f"  Loading on cuda:{i} ...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                token=token,
            ).to(f"cuda:{i}")
            model.eval()
            models.append(model)
    elif num_gpus == 1:
        print("Loading model on cuda:0 ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            token=token,
        ).to("cuda:0")
        model.eval()
        models.append(model)
    else:
        raise RuntimeError("No GPU found. This script requires at least one CUDA GPU.")

    print(f"✅ Models loaded on {len(models)} GPU(s).")
    return models, tokenizer


def build_prompt(text, tokenizer):
    """Build a chat-formatted prompt for translation."""
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nTranslate the following to Bengali:\n\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def generate_on_device(prompts, model, tokenizer, device):
    """Run generation for a list of prompts on a specific GPU."""
    if not prompts:
        return []

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    generated = outputs[:, input_length:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [t.strip() for t in decoded]


def translate_batch(texts, models, tokenizer):
    """Translate a batch of English texts to Bengali using all GPUs."""
    if not texts:
        return [""] * len(texts) if texts else []

    # Filter empties and track indices
    non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [""] * len(texts)

    # Build prompts
    prompts = [build_prompt(t, tokenizer) for t in non_empty_texts]

    # Distribute across GPUs
    if len(models) == 1:
        results = generate_on_device(prompts, models[0], tokenizer, models[0].device)
    else:
        chunk_size = (len(prompts) + len(models) - 1) // len(models)
        prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        tasks = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            for i, chunk in enumerate(prompt_chunks):
                if i < len(models):
                    model = models[i]
                    tasks.append(
                        executor.submit(generate_on_device, chunk, model, tokenizer, model.device)
                    )

        results = []
        for task in tasks:
            results.extend(task.result())

    # Map results back (fill empties)
    final = [""] * len(texts)
    for idx, result in zip(non_empty_indices, results):
        final[idx] = result

    return final


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Auth ──
    token = get_hf_token(args.hf_token)

    # ── Load model ──
    models, tokenizer = load_models(token)
    num_gpus = len(models)
    effective_batch_size = args.batch_size * num_gpus
    print(f"Using effective batch size of {effective_batch_size} across {num_gpus} GPU(s).")

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

    # ── Translate in batches ──
    mode = "a" if args.resume and start_idx > 0 else "w"
    batch_indices = list(range(start_idx, total, effective_batch_size))

    pbar = tqdm(
        batch_indices,
        desc="Translating",
        unit="batch",
        total=len(batch_indices),
    )

    with open(OUTPUT_FILE, mode, encoding="utf-8") as out_f:
        for i in pbar:
            batch_end = min(i + effective_batch_size, total)
            batch_rows = dataset[i:batch_end]

            instructions = batch_rows["instruction"]
            inputs_text = batch_rows["input"]
            outputs_text = batch_rows["output"]

            # Translate each field
            bn_instructions = translate_batch(instructions, models, tokenizer)
            bn_inputs = translate_batch(inputs_text, models, tokenizer)
            bn_outputs = translate_batch(outputs_text, models, tokenizer)

            # Write rows
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

            out_f.flush()

            # Save checkpoint
            with open(CHECKPOINT_FILE, "w") as ckpt_f:
                json.dump({"last_processed": batch_end}, ckpt_f)

            pbar.set_postfix({
                "done": f"{batch_end}/{total}",
                "samples/s": f"{(batch_end - start_idx) / pbar.format_dict['elapsed']:.1f}" if pbar.format_dict.get('elapsed', 0) > 0 else "...",
            })

    print(f"\n✅ Translation complete! Saved locally to: {OUTPUT_FILE}")
    print(f"   Total translated: {total - start_idx} rows")

    # ── Push to Hub ──
    if args.push_to_hub and args.hub_repo_id:
        print(f"\nUploading dataset to Hugging Face Hub ({args.hub_repo_id})...")
        try:
            final_dataset = load_dataset("json", data_files=OUTPUT_FILE, split="train")
            final_dataset.push_to_hub(
                args.hub_repo_id,
                token=args.hf_token or os.environ.get("HF_TOKEN"),
            )
            print(f"✅ Successfully pushed to https://huggingface.co/datasets/{args.hub_repo_id}")
        except Exception as e:
            print(f"⚠️ Failed to push to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
