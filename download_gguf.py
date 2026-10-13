"""Paste into one Colab cell after training, or run with Python to save locally.

Exports this notebook's GPT-2 + custom byte-level BPE to F16 GGUF, then starts a
browser download in Colab. No GPU, Drive mount, or llama.cpp checkout is needed.
GGUF stores weights AND the tokenizer; simply renaming safetensors will not work.
"""
from pathlib import Path
import json

# Settings — leave HF_REPO_ID blank to use the model in this Colab runtime.
MODEL_DIR = Path("female-authors/best_model")
HF_REPO_ID = ""  # Optional: "your-username/FemaleAuthorsScratch" instead of MODEL_DIR.
# For a private Hub model, provide HF_TOKEN via Colab Secrets or environment.
OUTPUT_FILE = Path("female-authors-f16.gguf")
DOWNLOAD_IN_COLAB = True


def export_gguf(model_dir=MODEL_DIR, output_file=OUTPUT_FILE):
    import gguf
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    model_dir, output_file = Path(model_dir), Path(output_file)
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("model_type") != "gpt2" or config.get("activation_function", "gelu_new") != "gelu_new"
            or config.get("add_cross_attention", False) or config.get("scale_attn_by_inverse_layer_idx", False)
            or not config.get("scale_attn_weights", True)):
        raise ValueError("This exporter supports the standard GPT-2 architecture used in main.py.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    pre, bpe = backend["pre_tokenizer"], backend["model"]
    # A fresh vocabulary cannot use llama.cpp's fingerprints of pretrained vocabularies.
    # Verify the actual GPT-2 pre-tokenization rules before declaring its GGUF type.
    if (not pre or pre.get("type") != "ByteLevel" or pre.get("add_prefix_space")
            or not pre.get("use_regex", True) or backend.get("normalizer") is not None
            or (backend.get("decoder") or {}).get("type") != "ByteLevel"
            or (backend.get("post_processor") or {}).get("type", "ByteLevel") != "ByteLevel"
            or bpe.get("type") != "BPE" or bpe.get("dropout")
            or any(bpe.get(key) for key in ("byte_fallback", "ignore_merges", "continuing_subword_prefix", "end_of_word_suffix"))):
        raise ValueError("Expected the unchanged ByteLevelBPETokenizer from main.py.")
    vocab = tokenizer.get_vocab()
    if len(tokenizer) != config["vocab_size"] or set(vocab.values()) != set(range(config["vocab_size"])):
        raise ValueError("Tokenizer IDs must match the model's complete embedding vocabulary.")
    if any(token is None for token in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)):
        raise ValueError("The exported tokenizer must define BOS, EOS, and padding tokens.")
    if any(config.get(name) != getattr(tokenizer, name) for name in ("bos_token_id", "eos_token_id", "pad_token_id")):
        raise ValueError("Model and tokenizer special token IDs disagree.")
    model, info = GPT2LMHeadModel.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True,
        dtype=torch.float32, output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete or incompatible model export: {info}")
    model.eval()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial = output_file.with_suffix(output_file.suffix + ".partial")
    writer = gguf.GGUFWriter(partial, "gpt2", use_temp_file=True)
    try:
        writer.add_name("FemaleAuthorsScratch")
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
        writer.add_block_count(model.config.n_layer)
        writer.add_context_length(model.config.n_positions)
        writer.add_embedding_length(model.config.n_embd)
        writer.add_feed_forward_length(model.config.n_inner or 4 * model.config.n_embd)
        writer.add_head_count(model.config.n_head)
        writer.add_layer_norm_eps(model.config.layer_norm_epsilon)
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        tokens = sorted(vocab, key=vocab.get)
        writer.add_token_list(tokens)
        special_ids = set(tokenizer.all_special_ids)
        writer.add_token_types([gguf.TokenType.CONTROL if i in special_ids else gguf.TokenType.NORMAL
                               for i in range(len(tokens))])
        merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in bpe["merges"]]
        if not merges:
            raise ValueError("The tokenizer has no learned BPE merges; train it before exporting.")
        writer.add_token_merges(merges)
        writer.add_bos_token_id(tokenizer.bos_token_id)
        writer.add_eos_token_id(tokenizer.eos_token_id)
        writer.add_pad_token_id(tokenizer.pad_token_id)
        writer.add_add_bos_token(False)  # Continuation prompts in main.py have no automatic BOS.
        writer.add_add_eos_token(False)

        # Use llama.cpp's tensor names; GPT-2's Conv1D projection matrices transpose.
        # Reference: https://github.com/ggml-org/llama.cpp/blob/master/conversion/gpt2.py
        names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GPT2, model.config.n_layer)
        for name, tensor in model.state_dict().items():
            if name == "lm_head.weight" and model.config.tie_word_embeddings:
                continue  # llama.cpp reuses the token embedding for the tied output head.
            mapped = names.get_name(name, try_suffixes=(".weight", ".bias"))
            if mapped is None:
                raise ValueError(f"Unsupported GPT-2 tensor: {name}")
            if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight")):
                tensor = tensor.T
            array = tensor.detach().cpu().float().numpy()
            dtype = np.float16 if array.ndim == 2 and name != "transformer.wpe.weight" else np.float32
            array = np.ascontiguousarray(array, dtype=dtype)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values after F16 conversion: {name}")
            writer.add_tensor(mapped, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        partial.replace(output_file)
    finally:
        writer.close()
        if writer.temp_file is not None:
            writer.temp_file.close()
        partial.unlink(missing_ok=True)
    print(f"Saved {output_file.resolve()} ({output_file.stat().st_size / 1024**2:.1f} MiB)")
    return output_file


def main():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf==0.17.1",
                    "transformers==5.2.0", "tokenizers==0.22.2", "safetensors>=0.4.3,<1"], check=True)
    model_dir = MODEL_DIR
    if HF_REPO_ID:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(HF_REPO_ID, allow_patterns=[
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "*.safetensors", "model.safetensors.index.json",
        ])
    path = export_gguf(model_dir, OUTPUT_FILE)
    if DOWNLOAD_IN_COLAB:
        try:
            from google.colab import files
        except ImportError:
            print("Outside Colab: the GGUF is available at the path above.")
        else:
            files.download(str(path))


if __name__ == "__main__":
    main()
