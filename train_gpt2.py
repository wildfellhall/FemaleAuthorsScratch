#!/usr/bin/env python3
"""Train a randomly initialized GPT2LMHeadModel on the audited corpus (single GPU)."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from corpus_registry import CorpusError, DEFAULT_SOURCES
import prepare_corpus
from prepare_corpus import sha256, write_json
from training_data import (DATA_VERSION, SPLITS, TokenWindows, collate_windows,
                           file_sha256, inspect_corpus, resolve_run,
                           tokenize_corpus, train_tokenizer)


# The preceding trainer differs only in progress sampling and its configuration.
# Allow its checkpoints after checking every training/data/environment setting.
PRE_SAMPLE_TRAINER_SHA256 = "e6bbb7324298a7187b8b9b5d605a642f4296a31fbb078d52e0750f64e881004a"


@dataclass
class TrainConfig:
    vocab_size: int = 32768
    min_frequency: int = 2
    context_length: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    batch_size: int = 8
    gradient_accumulation: int = 8
    epochs: int = 3
    max_steps: int = 0  # zero: derive the schedule from epochs and actual data size
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    eval_every: int = 200
    save_every: int = 100
    log_every: int = 10
    keep_checkpoints: int = 2  # best checkpoint is also retained
    gradient_checkpointing: bool = True
    seed: int = 42
    sample_every: int = 100  # optimizer updates, not accumulation microbatches; 0 disables
    sample_max_new_tokens: int = 128
    sample_temperature: float = 0.8
    sample_top_p: float = 0.95
    sample_seed: int = 1234
    sample_prompts: list[str] = field(default_factory=lambda: [
        "The evening was quiet, and",
        "She opened the letter and",
        '\"I never expected,\" she said,',
    ])

    def validate(self):
        integers = ("vocab_size", "min_frequency", "context_length", "n_layer", "n_head", "n_embd", "batch_size", "gradient_accumulation", "epochs", "eval_every", "save_every", "log_every", "keep_checkpoints")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in integers):
            raise ValueError("Model dimensions, batch sizes, epochs and intervals must be positive integers.")
        if self.vocab_size < 258 or self.context_length < 8:
            raise ValueError("Byte BPE needs at least 258 tokens; context_length must be >= 8.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if type(self.max_steps) is not int or self.max_steps < 0 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("max_steps must be a nonnegative integer and seed a nonnegative integer.")
        if not 0 <= self.dropout < 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("dropout and warmup_fraction must be in [0, 1).")
        if not all(math.isfinite(x) for x in (self.learning_rate, self.weight_decay, self.max_grad_norm)) or self.learning_rate <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer configuration.")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean.")
        if type(self.sample_every) is not int or self.sample_every < 0 or type(self.sample_seed) is not int or self.sample_seed < 0:
            raise ValueError("sample_every and sample_seed must be nonnegative integers.")
        if type(self.sample_max_new_tokens) is not int or self.sample_max_new_tokens < 1:
            raise ValueError("sample_max_new_tokens must be a positive integer.")
        if not math.isfinite(self.sample_temperature) or self.sample_temperature <= 0 or not 0 < self.sample_top_p <= 1:
            raise ValueError("sample_temperature must be finite and positive; sample_top_p must be in (0, 1].")
        if not isinstance(self.sample_prompts, list) or not self.sample_prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in self.sample_prompts):
            raise ValueError("sample_prompts must be a nonempty list of nonempty strings.")


def make_model(config: TrainConfig, tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel
    architecture = GPT2Config(
        vocab_size=len(tokenizer), n_positions=config.context_length,
        n_ctx=config.context_length, n_layer=config.n_layer,
        n_head=config.n_head, n_embd=config.n_embd,
        resid_pdrop=config.dropout, embd_pdrop=config.dropout,
        attn_pdrop=config.dropout, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=False, attn_implementation="sdpa", loss_type="ForCausalLM",
    )
    # This constructor initializes random weights. No pretrained model is loaded.
    return GPT2LMHeadModel(architecture)


def amp_context(device):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate(model, dataset, config, tokenizer, device):
    import torch
    if not len(dataset):
        return None
    model.eval()
    total_nll, targets = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(dataset), config.batch_size):
            batch = collate_windows([dataset[i] for i in range(start, min(start + config.batch_size, len(dataset)))], tokenizer.pad_token_id)
            count = int((batch["labels"][:, 1:] != -100).sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            with amp_context(device):
                output = model(**batch)
            loss = float(output.loss)
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss.")
            total_nll += loss * count
            targets += count
    loss = total_nll / targets
    return {"loss": loss, "perplexity": math.exp(loss) if loss < 700 else None, "target_tokens": targets}


def generate_samples(model, tokenizer, config: TrainConfig, device, step: int, output: Path):
    """Print/save comparable prose continuations without perturbing training RNG.

    Generation uses the current weights, evaluation mode, and no gradients. The
    caller's model mode and CPU/CUDA random states are restored even on failure.
    """
    import torch

    was_training = model.training
    samples = []
    new_tokens = min(config.sample_max_new_tokens, model.config.n_positions - 1)
    prompt_limit = model.config.n_positions - new_tokens
    # manual_seed seeds all CUDA generators as well as CPU, so fork all of them.
    cuda_devices = list(range(torch.cuda.device_count()))
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            for index, prompt in enumerate(config.sample_prompts):
                torch.manual_seed(config.sample_seed + index)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                truncated = len(prompt_ids) > prompt_limit
                prompt_ids = prompt_ids[-prompt_limit:] or [tokenizer.bos_token_id]
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with amp_context(device):
                    generated = model.generate(
                        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=new_tokens, do_sample=True,
                        temperature=config.sample_temperature, top_p=config.sample_top_p,
                        top_k=0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bad_words_ids=[[tokenizer.pad_token_id]],
                    )
                continuation_ids = generated[0, len(prompt_ids):].tolist()
                samples.append({
                    "prompt": prompt, "input_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    "prompt_truncated": truncated,
                    "continuation": tokenizer.decode(continuation_ids, skip_special_tokens=True),
                    "generated_tokens": len(continuation_ids), "seed": config.sample_seed + index,
                })
    finally:
        model.train(was_training)
    record = {"step": step, "max_new_tokens": new_tokens,
              "temperature": config.sample_temperature, "top_p": config.sample_top_p,
              "top_k": 0, "samples": samples}
    with (output / "samples.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"\nSamples at optimizer step {step}", flush=True)
    for index, sample in enumerate(samples, 1):
        notice = " (prompt shortened to fit context)" if sample["prompt_truncated"] else ""
        print(f"\nPrompt {index}{notice}: {sample['input_text']}\nContinuation: {sample['continuation']}", flush=True)
    return record


def resume_recipe_matches(saved: dict, current: dict) -> bool:
    """Accept the verified pre-sampling trainer without relaxing other guards."""
    previous = dict(saved)
    implementation = dict(previous.get("implementation", {}))
    if implementation.get("train_gpt2.py") == PRE_SAMPLE_TRAINER_SHA256:
        implementation["train_gpt2.py"] = current["implementation"]["train_gpt2.py"]
    previous["implementation"] = implementation
    return previous == current


def checkpoint_paths(output):
    paths = [path for path in output.glob("checkpoint-*") if path.is_dir() and (path / "complete.json").is_file() and path.name.removeprefix("checkpoint-").isdigit()]
    return sorted(paths, key=lambda p: int(p.name.split("-")[-1]))


def save_checkpoint(output, model, optimizer, scheduler, state, identity, config):
    import numpy as np
    import torch
    target = output / f"checkpoint-{state['step']}"
    if target.exists():
        raise CorpusError(f"Checkpoint already exists: {target}")
    staging = output / f".checkpoint-{state['step']}.incomplete"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "state": state, "identity": identity, "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, staging / "training_state.pt")
    write_json(staging / "complete.json", {"step": state["step"], "identity": identity})
    staging.rename(target)
    # Delete only checkpoints generated by this run, after the new one is complete.
    paths = checkpoint_paths(output)
    keep = set(paths[-config.keep_checkpoints:])
    if state["best_step"] is not None:
        keep.add(output / f"checkpoint-{state['best_step']}")
    for path in paths:
        if path not in keep:
            shutil.rmtree(path)
    print(f"Checkpoint saved: {target}", flush=True)


def restore_checkpoint(path, model, optimizer, scheduler, identity, device):
    import numpy as np
    import torch
    from safetensors.torch import load_model
    marker = json.loads((path / "complete.json").read_text())
    if marker["identity"] != identity:
        raise CorpusError("Checkpoint identity does not match this run.")
    load_model(model, str(path / "model.safetensors"), device=str(device))
    # This pickle is created by this script inside the selected run directory.
    # Never resume an untrusted third-party training_state.pt file.
    saved = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if saved["identity"] != identity:
        raise CorpusError("Training state identity mismatch.")
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda" and saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["state"]


def log_metrics(output, values):
    print(json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as target:
        target.write(json.dumps(values, allow_nan=False) + "\n")


def train(args, config):
    import numpy as np
    import torch
    from transformers import PreTrainedTokenizerFast

    config.validate()
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise CorpusError("No CUDA GPU. In Colab choose Runtime > Change runtime type > A100 GPU. Use --device cpu only for a tiny smoke test.")
        if not torch.cuda.is_bf16_supported():
            raise CorpusError("This training setup requires a BF16-capable GPU such as A100.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        hardware = torch.cuda.get_device_name(0)
    else:
        hardware = "CPU (smoke tests only)"
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Device: {hardware}", flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    run = resolve_run(args.corpus)
    corpus = inspect_corpus(run, args.allow_partial_corpus)
    settings = asdict(config)
    sampling_settings = {key: value for key, value in settings.items() if key.startswith("sample_")}
    training_settings = {key: value for key, value in settings.items() if not key.startswith("sample_")}
    recipe = {"config": training_settings, "data_version": DATA_VERSION, "corpus": corpus["fingerprint"],
              "precision": "bf16" if device.type == "cuda" else "fp32",
              "implementation": {name: file_sha256(Path(__file__).with_name(name)) for name in ("train_gpt2.py", "training_data.py")},
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "numpy")}}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_file = output / "run.json"
    if run_file.exists():
        if not args.resume:
            raise CorpusError("Output already contains a run. Use --resume or a new --output directory.")
        saved_run = json.loads(run_file.read_text())
        if not resume_recipe_matches(saved_run["recipe"], recipe):
            raise CorpusError("Corpus, settings, precision or library versions changed. Use a new output directory; resume cannot change the training objective or schedule.")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(output / "tokenizer", local_files_only=True)
        if {p.name: file_sha256(p) for p in (output / "tokenizer").iterdir() if p.is_file()} != saved_run["tokenizer_files"]:
            raise CorpusError("Saved tokenizer changed; refusing to resume.")
    else:
        if any(output.iterdir()):
            raise CorpusError("Output is not empty and has no run.json. Choose a fresh directory.")
        if args.resume:
            raise CorpusError("No run to resume. Omit --resume for the first session.")
        tokenizer = train_tokenizer(run, output / "tokenizer", config.vocab_size, config.min_frequency)
        tokenizer_files = {p.name: file_sha256(p) for p in (output / "tokenizer").iterdir() if p.is_file()}
        tokenizer_hash = sha256(json.dumps(tokenizer_files, sort_keys=True).encode())
        saved_run = {"recipe": recipe, "tokenizer_sha256": tokenizer_hash, "tokenizer_files": tokenizer_files, "corpus": corpus,
                     "identity": sha256(json.dumps({"recipe": recipe, "tokenizer": tokenizer_hash}, sort_keys=True).encode())}
        provenance = output / "provenance"
        provenance.mkdir()
        for name in ("train_gpt2.py", "training_data.py", "corpus_registry.py", "prepare_corpus.py"):
            shutil.copy2(Path(__file__).with_name(name), provenance / name)
        for name in ("report.json", "manifest.jsonl"):
            shutil.copy2(run / name, provenance / name)
        cleaning_report = json.loads((run / "report.json").read_text())
        for name, fallback, checksum in (("corpus-list.py", args.sources, "sources_sha256"), ("cleaning_rules.json", args.overrides, "rules_sha256")):
            source = run / name if (run / name).is_file() else fallback
            if source.is_file() and file_sha256(source) == cleaning_report[checksum]:
                shutil.copy2(source, provenance / name)
        write_json(run_file, saved_run)
    write_json(output / "sampling_config.json", sampling_settings)
    identity = saved_run["identity"]
    token_dir = args.work_dir / identity
    marker = token_dir / "complete.json"
    if not marker.exists():
        # If Colab lost /content, rebuild tokens with the frozen tokenizer on Drive.
        stats = tokenize_corpus(run, token_dir, tokenizer)
        write_json(marker, {"identity": identity, "index_sha256": file_sha256(token_dir / "index.json")})
    else:
        completion = json.loads(marker.read_text())
        if completion["identity"] != identity or completion["index_sha256"] != file_sha256(token_dir / "index.json"):
            raise CorpusError("Token cache metadata changed. Choose a fresh --work-dir to rebuild.")
        stats = json.loads((token_dir / "index.json").read_text())
        for split in SPLITS:
            if file_sha256(token_dir / f"{split}.bin") != stats[split]["sha256"]:
                raise CorpusError(f"Corrupt {split} token cache. Choose a fresh --work-dir to rebuild.")
    datasets = {split: TokenWindows(token_dir, split, config.context_length) for split in SPLITS}
    batches_per_epoch = math.ceil(len(datasets["train"]) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps:
        total_steps = min(total_steps, config.max_steps)
    warmup_steps = min(max(1, round(total_steps * config.warmup_fraction)), total_steps - 1) if config.warmup_fraction else 0

    model = make_model(config, tokenizer).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")

    def lr_scale(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    state = {"epoch": 0, "next_batch": 0, "step": 0, "tokens_seen": 0, "best_loss": None, "best_step": None}
    checkpoints = checkpoint_paths(output)
    if args.resume and checkpoints:
        state = restore_checkpoint(checkpoints[-1], model, optimizer, scheduler, identity, device)
        print(f"Resuming optimizer step {state['step']}.", flush=True)
    elif args.resume:
        print("No complete checkpoint yet; restarting this initialized run from its seed.", flush=True)
        # Tokenizer training may use its own RNG; model initialization uses the seed above.
    summary = {"parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(tokenizer),
               "splits": {s: {"books": len(stats[s]["books"]), "target_tokens": stats[s]["target_tokens"], "windows": len(datasets[s])} for s in SPLITS},
               "planned_optimizer_steps": total_steps, "steps_per_epoch": steps_per_epoch,
               "maximum_tokens_per_update": config.batch_size * config.gradient_accumulation * (config.context_length - 1), "hardware": hardware}
    write_json(output / "training_plan.json", summary)
    print(json.dumps(summary, indent=2), flush=True)

    started, session_tokens = time.monotonic(), 0
    stop_reason = "complete"
    while state["step"] < total_steps and state["epoch"] < config.epochs:
        epoch = state["epoch"]
        order = np.random.default_rng(config.seed + epoch).permutation(len(datasets["train"]))
        for batch_number in range(state["next_batch"], batches_per_epoch, config.gradient_accumulation):
            batch_end = min(batch_number + config.gradient_accumulation, batches_per_epoch)
            microbatches = []
            for number in range(batch_number, batch_end):
                indices = order[number * config.batch_size:(number + 1) * config.batch_size]
                microbatches.append(collate_windows([datasets["train"][int(i)] for i in indices], tokenizer.pad_token_id))
            targets = [int((batch["labels"][:, 1:] != -100).sum()) for batch in microbatches]
            total_targets = sum(targets)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            nll = 0.0
            for batch, count in zip(microbatches, targets):
                batch = {key: value.to(device) for key, value in batch.items()}
                with amp_context(device):
                    result = model(**batch)
                    # Weight by real targets, including a partial final accumulation.
                    loss = result.loss * (count / total_targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; last complete checkpoint remains available.")
                nll += float(result.loss.detach()) * count
                loss.backward()
                del result, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            state["tokens_seen"] += total_targets
            session_tokens += total_targets
            state["next_batch"] = batch_end
            if batch_end == batches_per_epoch:
                state["epoch"], state["next_batch"] = epoch + 1, 0
            if config.sample_every and state["step"] % config.sample_every == 0:
                generate_samples(model, tokenizer, config, device, state["step"], output)
            elapsed = time.monotonic() - started
            if state["step"] == 1 or state["step"] % config.log_every == 0:
                log_metrics(output, {"step": state["step"], "train_loss": nll / total_targets, "learning_rate": scheduler.get_last_lr()[0], "tokens_seen": state["tokens_seen"], "session_seconds": elapsed, "session_tokens_per_second": session_tokens / max(elapsed, 1e-9)})
            finished = state["step"] >= total_steps or state["epoch"] >= config.epochs
            timed_out = args.max_seconds > 0 and elapsed >= args.max_seconds
            should_eval = state["step"] % config.eval_every == 0 or batch_end == batches_per_epoch or finished or timed_out
            if should_eval:
                metrics = evaluate(model, datasets["validation"], config, tokenizer, device)
                log_metrics(output, {"step": state["step"], "validation": metrics})
                if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
                    state["best_loss"], state["best_step"] = metrics["loss"], state["step"]
            if should_eval or state["step"] % config.save_every == 0 or finished or timed_out:
                save_checkpoint(output, model, optimizer, scheduler, state, identity, config)
            if finished or timed_out:
                stop_reason = "complete" if finished else "session_time_limit"
                break
        if stop_reason == "session_time_limit":
            break

    # Export the validation-selected model; test is touched only at schedule completion.
    from safetensors.torch import load_model
    best = output / f"checkpoint-{state['best_step']}"
    load_model(model, str(best / "model.safetensors"), device=str(device))
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.gradient_checkpointing_disable()
    model.save_pretrained(output / "best_model", safe_serialization=True)
    tokenizer.model_max_length = config.context_length
    tokenizer.save_pretrained(output / "best_model")
    complete = state["step"] >= total_steps or state["epoch"] >= config.epochs
    result = {"complete": complete, "stop_reason": stop_reason, "step": state["step"], "best_step": state["best_step"], "best_validation_loss": state["best_loss"], "tokens_seen": state["tokens_seen"], "session_seconds": time.monotonic() - started, "model": str(output / "best_model")}
    if complete:
        model.config.use_cache = False
        result["test"] = evaluate(model, datasets["test"], config, tokenizer, device)
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"), help="Preparation root or exact run directory")
    parser.add_argument("--output", type=Path, default=Path("runs/gpt2"), help="Persistent run/checkpoint directory; use Google Drive in Colab")
    parser.add_argument("--work-dir", type=Path, default=Path("data/tokenized"), help="Local token cache; may be rebuilt on resume")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("training_config.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume latest complete checkpoint in --output")
    parser.add_argument("--max-seconds", type=float, default=7200, help="Session training budget; checked after updates. Evaluation/checkpoint/export can run longer. 0 disables.")
    parser.add_argument("--max-steps", type=int, help="Override planned total optimizer steps (not extra steps on resume)")
    parser.add_argument("--allow-partial-corpus", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="First download/clean the current corpus list with raw caching")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--overrides", type=Path, default=Path(__file__).with_name("cleaning_rules.json"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("--max-seconds must be finite and nonnegative.")
    if args.prepare and args.resume:
        parser.error("Resume reuses its existing prepared corpus; do not combine --prepare and --resume.")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        config = TrainConfig(**json.loads(args.config.read_text()))
        if args.max_steps is not None:
            config.max_steps = args.max_steps
        config.validate()
        if args.prepare:
            previous = (args.corpus / "latest.json").read_bytes() if (args.corpus / "latest.json").exists() else None
            status = prepare_corpus.main(["--sources", str(args.sources), "--overrides", str(args.overrides), "--output", str(args.corpus), "--cache-raw"])
            latest = args.corpus / "latest.json"
            if not latest.exists() or latest.read_bytes() == previous:
                raise CorpusError("Preparation did not produce a new run; refusing to train on stale outputs.")
            if status and not args.allow_partial_corpus:
                raise CorpusError("Preparation needs review. Inspect its audit before training, or explicitly use --allow-partial-corpus.")
        train(args, config)
    except (CorpusError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
