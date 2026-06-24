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
        if not hasattr(instance, "tie_weights"):
            return None
        orig_tie = getattr(instance, "tie_weights")
        import functools
        import inspect
        try:
            sig = inspect.signature(orig_tie)
        except (ValueError, TypeError):
            return orig_tie
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
            if orig_tie is not None and hasattr(self, 'tie_weights') and getattr(self, 'tie_weights', None) != orig_tie:
                del self.tie_weights
    transformers.modeling_utils.PreTrainedModel.init_weights = custom_init_weights
    
    # Also patch _finalize_model_loading as it calls tie_weights directly
    finalize_desc = transformers.modeling_utils.PreTrainedModel.__dict__.get("_finalize_model_loading")
    if finalize_desc is not None:
        if isinstance(finalize_desc, classmethod):
            orig_func = finalize_desc.__func__
            def custom_finalize_func(cls, model, *args, **kwargs):
                orig_tie = apply_instance_tie_weights_wrapper(model)
                try:
                    return orig_func(cls, model, *args, **kwargs)
                finally:
                    if orig_tie is not None and hasattr(model, 'tie_weights') and getattr(model, 'tie_weights', None) != orig_tie:
                        del model.tie_weights
            transformers.modeling_utils.PreTrainedModel._finalize_model_loading = classmethod(custom_finalize_func)
        else:
            orig_func = finalize_desc
            def custom_finalize_func(cls, model, *args, **kwargs):
                orig_tie = apply_instance_tie_weights_wrapper(model)
                try:
                    return orig_func(cls, model, *args, **kwargs)
                finally:
                    if orig_tie is not None and hasattr(model, 'tie_weights') and getattr(model, 'tie_weights', None) != orig_tie:
                        del model.tie_weights
            transformers.modeling_utils.PreTrainedModel._finalize_model_loading = custom_finalize_func
    
    # 3. Mock removed transformers.onnx module
    onnx_mock = ModuleType("transformers.onnx")
    class DummyConfig: pass
    onnx_mock.OnnxConfig = DummyConfig
    onnx_mock.OnnxConfigWithPast = DummyConfig
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
    
    # 5. Patch Cache objects for older models that expect past_key_values to be a tuple
    try:
        import transformers.cache_utils
        def _cache_getitem(self, idx):
            if hasattr(self, "key_cache") and hasattr(self, "value_cache"):
                return (self.key_cache[idx], self.value_cache[idx])
            raise KeyError(idx)
        def _cache_len(self):
            if hasattr(self, "key_cache"):
                return len(self.key_cache)
            return 0
        def _cache_iter(self):
            for i in range(len(self)):
                yield self[i]
        for cls_name in ["Cache", "DynamicCache", "EncoderDecoderCache"]:
            if hasattr(transformers.cache_utils, cls_name):
                cls = getattr(transformers.cache_utils, cls_name)
                if not hasattr(cls, "__getitem__"):
                    cls.__getitem__ = _cache_getitem
                if not hasattr(cls, "__len__"):
                    cls.__len__ = _cache_len
                if not hasattr(cls, "__iter__"):
                    cls.__iter__ = _cache_iter
    except Exception:
        pass
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
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the translated dataset to Hugging Face Hub when complete")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                        help="Hugging Face repo ID to push to (e.g. username/dataset_name)")
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
    ip = IndicProcessor(inference=True)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    models = []
    
    if num_gpus > 1:
        print(f"Found {num_gpus} GPUs. Loading model on each GPU for data parallelism...")
        for i in range(num_gpus):
            print(f"Loading model on cuda:{i} ...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=token
            ).to(f"cuda:{i}")
            model.eval()
            models.append(model)
    else:
        print(f"Loading model on {DEVICE} ...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=token
        ).to(DEVICE)
        model.eval()
        models.append(model)

    print("Models loaded successfully!")
    return models, tokenizer, ip


def _generate_only(inputs, model):
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=MAX_LENGTH,
            num_beams=5,
            num_return_sequences=1,
        )
    return generated_tokens


def translate_batch(sentences, models, tokenizer, ip):
    """Translate a batch of English sentences to Bengali."""
    if not sentences:
        return []

    # 1. Pre-process and Tokenize sequentially in the main thread
    batch = ip.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        max_length=MAX_LENGTH,
        return_tensors="pt",
        return_attention_mask=True
    )

    # 2. Distribute only the generation across GPUs
    batch_size = len(sentences)
    if len(models) == 1:
        inputs_device = {k: v.to(models[0].device) for k, v in inputs.items()}
        generated_tokens = _generate_only(inputs_device, models[0])
    else:
        from concurrent.futures import ThreadPoolExecutor
        chunk_size = (batch_size + len(models) - 1) // len(models)
        
        input_chunks = []
        for i in range(0, batch_size, chunk_size):
            chunk_inputs = {k: v[i:i+chunk_size] for k, v in inputs.items()}
            input_chunks.append(chunk_inputs)
            
        tasks = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            for i, chunk_inputs in enumerate(input_chunks):
                if i < len(models):
                    model = models[i]
                    chunk_inputs_device = {k: v.to(model.device) for k, v in chunk_inputs.items()}
                    tasks.append(executor.submit(_generate_only, chunk_inputs_device, model))
        decoded = []
        for task in tasks:
            chunk_tokens = task.result().cpu()
            decoded.extend(tokenizer.batch_decode(chunk_tokens, skip_special_tokens=True))

    # 3. Post-process sequentially in the main thread
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
    models, tokenizer, ip = load_translation_model(hf_token=args.hf_token)
    num_gpus = len(models)
    effective_batch_size = args.batch_size * num_gpus
    print(f"Using effective batch size of {effective_batch_size} across {num_gpus} GPU(s).")

    # ── Translate in batches ──
    mode = "a" if args.resume and start_idx > 0 else "w"
    with open(OUTPUT_FILE, mode, encoding="utf-8") as out_f:
        for i in range(start_idx, total, effective_batch_size):
            batch_end = min(i + effective_batch_size, total)
            batch_rows = dataset[i:batch_end]

            # Collect texts to translate
            instructions = batch_rows["instruction"]
            inputs_text = batch_rows["input"]
            outputs_text = batch_rows["output"]

            # Translate each field
            bn_instructions = translate_batch(instructions, models, tokenizer, ip)
            bn_inputs = translate_batch(inputs_text, models, tokenizer, ip)
            bn_outputs = translate_batch(outputs_text, models, tokenizer, ip)

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

    print(f"\n✅ Translation complete! Saved locally to: {OUTPUT_FILE}")
    print(f"   Total translated: {total - start_idx} rows")

    if args.push_to_hub and args.hub_repo_id:
        print(f"\nUploading dataset to Hugging Face Hub ({args.hub_repo_id})...")
        try:
            final_dataset = load_dataset("json", data_files=OUTPUT_FILE, split="train")
            final_dataset.push_to_hub(args.hub_repo_id, token=args.hf_token or os.environ.get("HF_TOKEN"))
            print(f"✅ Successfully pushed to https://huggingface.co/datasets/{args.hub_repo_id}")
        except Exception as e:
            print(f"⚠️ Failed to push to Hugging Face Hub: {e}")


if __name__ == "__main__":
    main()
