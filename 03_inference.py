"""
Step 3: Run inference with the fine-tuned Bengali Medical Gemma model.
Uses Unsloth's FastModel for optimized inference.

Usage:
    python 03_inference.py
    python 03_inference.py --adapter_path output/gemma-1b-bengali-medical
    python 03_inference.py --merged_path output/gemma-1b-bengali-medical/merged_16bit
"""

import argparse
import torch
from transformers import TextStreamer

# ─────────────────────────────── Config ───────────────────────────────

BASE_MODEL = "unsloth/gemma-3-1b-it"
ADAPTER_PATH = "output/gemma-1b-bengali-medical"
MAX_SEQ_LENGTH = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="Bengali Medical Chatbot Inference (Unsloth)")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH,
                        help="Path to LoRA adapter (default)")
    parser.add_argument("--merged_path", type=str, default=None,
                        help="Path to merged model (if saved with save_pretrained_merged)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--stream", action="store_true", default=True,
                        help="Stream output token by token")
    return parser.parse_args()


def load_model_unsloth(base_model, adapter_path, merged_path=None):
    """Load model using Unsloth's FastModel for optimized inference."""
    from unsloth import FastModel

    if merged_path:
        # Load the fully merged model directly
        print(f"Loading merged model: {merged_path}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=merged_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=None,
        )
    else:
        # Load base + LoRA adapter
        print(f"Loading base model: {base_model}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=base_model,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=None,
        )
        print(f"Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    # Enable optimized inference mode
    FastModel.for_inference(model)
    return model, tokenizer


def generate_response(model, tokenizer, question, max_new_tokens=512, stream=True):
    """Generate a Bengali medical response using Gemma chat template."""
    messages = [
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    if stream:
        # Stream tokens one by one
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            use_cache=True,
        )
        return None  # Already printed via streamer
    else:
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            use_cache=True,
        )
        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()


def main():
    args = parse_args()
    model, tokenizer = load_model_unsloth(
        args.base_model, args.adapter_path, args.merged_path
    )

    print("\n" + "=" * 60)
    print("  🩺 বাংলা মেডিকেল চ্যাটবট (Bengali Medical Chatbot)")
    print("  Powered by Gemma 1B + Unsloth")
    print("  Type your question in Bengali. Type 'quit' to exit.")
    print("=" * 60)

    # Example prompts
    examples = [
        "আমার মাথা ব্যথা এবং জ্বর হচ্ছে। আমি কি করব?",
        "ডায়াবেটিস রোগীদের জন্য কোন খাবার ভালো?",
        "আমার বুকে ব্যথা হচ্ছে, এটা কি হার্ট অ্যাটাকের লক্ষণ?",
    ]
    print("\n📋 Example questions you can try:")
    for i, ex in enumerate(examples, 1):
        print(f"   {i}. {ex}")
    print()

    while True:
        question = input("🤒 আপনার প্রশ্ন: ").strip()
        if question.lower() in ("quit", "exit", "বাদ"):
            print("ধন্যবাদ! সুস্থ থাকুন। 🙏")
            break
        if not question:
            continue

        print("\n👨‍⚕️ ডাক্তার: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=args.max_new_tokens,
            stream=args.stream,
        )
        if response:  # Only print if not streaming
            print(response)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
