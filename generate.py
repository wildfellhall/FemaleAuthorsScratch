#!/usr/bin/env python3
"""Generate text from a locally trained best_model export; no Hub downloads."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The evening was quiet, and")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Token limit and temperature must be positive; top-p must be in (0, 1].")
    import torch
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, set_seed
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    model.config.use_cache = True
    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    if inputs["input_ids"].shape[1] == 0:
        inputs = {"input_ids": torch.tensor([[tokenizer.bos_token_id]]), "attention_mask": torch.ones((1, 1), dtype=torch.long)}
    if inputs["input_ids"].shape[1] + args.max_new_tokens > model.config.n_positions:
        parser.error(f"Prompt plus generated tokens must fit {model.config.n_positions} tokens. Shorten the prompt or --max-new-tokens.")
    with torch.inference_mode():
        result = model.generate(**{k: v.to(device) for k, v in inputs.items()}, max_new_tokens=args.max_new_tokens, use_cache=True, do_sample=True, temperature=args.temperature, top_p=args.top_p, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bad_words_ids=[[tokenizer.pad_token_id]])
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
