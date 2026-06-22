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
# Workaround for compatibility between newer transformers versions and IndicTransToolkit
try:
    import sys
    from types import ModuleType
    import transformers.tokenization_utils_base
    import transformers.tokenization_utils
    import transformers.dynamic_module_utils
    
    # 1. Patch PreTrainedTokenizerBase
    transformers.tokenization_utils.PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase
    sys.modules['transformers.tokenization_utils'].PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase
    
    # Patch __getattr__ on PreTrainedTokenizerBase for _special_tokens_map
    orig_getattr = transformers.tokenization_utils_base.PreTrainedTokenizerBase.__getattr__
    def new_getattr(self, name):
        if name == "_special_tokens_map":
            self.__dict__["_special_tokens_map"] = {}
            return self.__dict__["_special_tokens_map"]
        return orig_getattr(self, name)
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.__getattr__ = new_getattr
    
    # 2. Patch dynamic module loading to wrap tie_weights if overridden
    orig_get_class = transformers.dynamic_module_utils.get_class_from_dynamic_module
    def custom_get_class(*args, **kwargs):
        cls = orig_get_class(*args, **kwargs)
        if hasattr(cls, "tie_weights"):
            orig_tie = cls.tie_weights
            import functools
            import inspect
            @functools.wraps(orig_tie)
            def wrapped_tie(self, *args, **kwargs):
                sig = inspect.signature(orig_tie)
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    allowed_keys = set(sig.parameters.keys())
                    kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
                return orig_tie(self, *args, **kwargs)
            cls.tie_weights = wrapped_tie
        return cls
    
    transformers.dynamic_module_utils.get_class_from_dynamic_module = custom_get_class
    try:
        import transformers.models.auto.auto_factory
        transformers.models.auto.auto_factory.get_class_from_dynamic_module = custom_get_class
    except Exception: pass
    try:
        import transformers.models.auto.configuration_auto
        transformers.models.auto.configuration_auto.get_class_from_dynamic_module = custom_get_class
    except Exception: pass
    try:
        import transformers.models.auto.tokenization_auto
        transformers.models.auto.tokenization_auto.get_class_from_dynamic_module = custom_get_class
    except Exception: pass
    
    # 2.b Fail-safe: Patch PreTrainedModel.init_weights and related tie_weights calls to wrap dynamically on instances
    import transformers.modeling_utils
    
    def apply_instance_tie_weights_wrapper(instance):
        orig_tie = instance.tie_weights
        import functools
        import inspect
        sig = inspect.signature(orig_tie)
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            allowed_keys = set(sig.parameters.keys())
            @functools.wraps(orig_tie)
            def temp_tie(*args, **kwargs):
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
                return orig_tie(*args, **filtered_kwargs)
            instance.tie_weights = temp_tie
        return orig_tie

    orig_init_weights = transformers.modeling_utils.PreTrainedModel.init_weights
    def custom_init_weights(self):
        orig_tie = apply_instance_tie_weights_wrapper(self)
        try:
            orig_init_weights(self)
        finally:
            if hasattr(self, 'tie_weights') and self.tie_weights != orig_tie:
                del self.tie_weights
    transformers.modeling_utils.PreTrainedModel.init_weights = custom_init_weights
    
    # Also patch _finalize_model_loading as it calls tie_weights directly
    if hasattr(transformers.modeling_utils.PreTrainedModel, "_finalize_model_loading"):
        orig_finalize = transformers.modeling_utils.PreTrainedModel._finalize_model_loading
        @classmethod
        def custom_finalize(cls, model, *args, **kwargs):
            orig_tie = apply_instance_tie_weights_wrapper(model)
            try:
                return orig_finalize.__func__(cls, model, *args, **kwargs)
            finally:
                if hasattr(model, 'tie_weights') and model.tie_weights != orig_tie:
                    del model.tie_weights
        transformers.modeling_utils.PreTrainedModel._finalize_model_loading = custom_finalize
    
    # 3. Mock removed transformers.onnx module
    onnx_mock = ModuleType("transformers.onnx")
    class DummyConfig: pass
    onnx_mock.OnnxConfig = DummyConfig
    onnx_mock.OnnxSeq2SeqConfigWithPast = DummyConfig
    sys.modules["transformers.onnx"] = onnx_mock
    
    # 4. Mock transformers.onnx.utils submodule
    onnx_utils_mock = ModuleType("transformers.onnx.utils")
    def compute_effective_axis_dimension(*args, **kwargs):
        pass
    onnx_utils_mock.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = onnx_utils_mock
    
    # Enable attribute access: transformers.onnx.utils
    onnx_mock.utils = onnx_utils_mock
except Exception:
    pass

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
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated repositories")
    return parser.parse_args()


def load_translation_model(hf_token=None):
    """Load IndicTrans2 model, tokenizer, and processor."""
    token = hf_token or os.environ.get("HF_TOKEN")
    
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

    print(f"Loading IndicTrans2 model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        token=token
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
    model, tokenizer, ip = load_translation_model(hf_token=args.hf_token)

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
