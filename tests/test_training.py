"""Exercise the notebook's custom PyTorch loop with tiny, invented CPU fixtures."""
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from test_prepare_corpus import CELLS, NARRATIVE, book_bytes, corpus

HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers", "numpy"))


@unittest.skipUnless(HAS_TRAINING, "Install the dependencies in main.py Cell 1 for ML tests")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.notebook = types.ModuleType("notebook_training")
        sys.modules[self.notebook.__name__] = self.notebook
        self.addCleanup(sys.modules.pop, self.notebook.__name__, None)
        scope = self.notebook.__dict__
        with contextlib.chdir(self.root), patch.object(subprocess, "run") as install:
            exec(CELLS[1], scope)
        install.assert_called_once()
        self.assertIn("pip", install.call_args.args[0])
        scope["torch"].set_num_threads(1)
        scope["OUTPUT_DIR"] = self.root / "female-authors"
        scope["DEVICE"] = "cpu"
        scope["MAX_SESSION_SECONDS"] = 0
        scope["config"] = replace(scope["config"], vocab_size=300, min_frequency=1,
            context_length=32, n_layer=1, n_head=2, n_embd=16, batch_size=2,
            gradient_accumulation=2, max_steps=3, epochs=2, eval_every=2,
            save_every=1, log_every=10, sample_every=1, sample_max_new_tokens=2,
            sample_prompts=["Mira opened"], seed=7)
        exec(CELLS[2], scope)
        exec(CELLS[3], scope)
        sources, self.raw = [], {}
        for index, split in enumerate(("train", "validation", "test")):
            book_id = next(str(i) for i in range(1, 10000)
                           if corpus.split_for(str(i), "gutenberg-corpus-v1", .05, .05) == split)
            sources.append(f"https://www.gutenberg.org/ebooks/{book_id}")
            text = NARRATIVE.replace("Mira", ("Mira", "Valeria Ω", "Zelda Ж")[index])
            self.raw[book_id] = book_bytes(text, book_id)
        scope["sources"] = sources
        def fetch(book):
            raw = self.raw[book.id]
            return raw, {"sha256": corpus.sha256(raw)}
        with patch.object(scope["Downloader"], "fetch", side_effect=fetch), contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS[4], scope)
        exec(CELLS[5], scope)
        # Define train without starting the notebook's final training call yet.
        exec(CELLS[6].split("\nmodel, result = train(")[0], scope)

    def run_training(self, output, config=None, resume=False, seconds=0):
        n = self.notebook
        with contextlib.redirect_stdout(io.StringIO()):
            return n.train(n.datasets, n.tokenizer, config or n.config, output,
                           resume=resume, max_seconds=seconds, device_name="cpu")

    def test_no_hugging_face_trainer_import(self):
        names = self.notebook.__dict__
        self.assertNotIn("Trainer", names)
        self.assertNotIn("TrainingArguments", names)
        self.assertNotIn("TrainerCallback", names)
        self.assertEqual(names["TrainConfig"]().sample_every, 100)

    def test_tokenizer_only_sees_training_books_and_preserves_unicode(self):
        n = self.notebook
        backend = n.ByteLevelBPETokenizer()
        backend.train_from_iterator(n.books_by_split["train"], vocab_size=300,
            min_frequency=1, special_tokens=n.SPECIAL_TOKENS, show_progress=False)
        self.assertEqual(n.tokenizer.get_vocab(), backend.get_vocab())
        text = "Ω Ж café — a test."
        self.assertEqual(n.tokenizer.decode(n.tokenizer.encode(text)), text)

    def test_windows_cover_every_target_and_never_cross_books(self):
        n = self.notebook
        texts = ["Mira " * 40, "Evening " * 23]
        data = n.TokenWindows(texts, n.tokenizer, 17)
        offset = 0
        for book, end in zip(data.books, data.ends):
            reconstructed = []
            for index in range(offset, end):
                reconstructed.extend(data[index][1:])
            self.assertEqual(reconstructed, book[1:].tolist())
            self.assertEqual(data[end - 1][-1], n.tokenizer.eos_token_id)
            offset = end
        with self.assertRaises(IndexError):
            data[len(data)]

    def test_padding_masks_only_padding_and_eos_is_trained(self):
        n = self.notebook
        eos, pad = n.tokenizer.eos_token_id, n.tokenizer.pad_token_id
        batch = n.collate_windows([[eos, 10, eos], [eos, eos]], pad)
        self.assertNotEqual(eos, pad)
        self.assertEqual(batch["labels"].tolist(), [[eos, 10, eos], [eos, eos, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])

    def test_samples_preserve_rng_mode_and_fit_context(self):
        n = self.notebook
        model = n.make_model(n.config, n.tokenizer).train()
        config = replace(n.config, sample_prompts=["A very long prompt " * 40], sample_max_new_tokens=8)
        before = n.torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()):
            first = n.generate_samples(model, n.tokenizer, config, n.torch.device("cpu"), 100, self.root)
            second = n.generate_samples(model, n.tokenizer, config, n.torch.device("cpu"), 200, self.root)
        self.assertTrue(model.training)
        self.assertTrue(n.torch.equal(before, n.torch.get_rng_state()))
        self.assertEqual(first["samples"], second["samples"])
        self.assertTrue(first["samples"][0]["prompt_truncated"])
        self.assertLessEqual(first["samples"][0]["generated_tokens"], 8)
        self.assertFalse(model.config.use_cache)

    def test_resume_matches_uninterrupted_and_rejects_changed_data(self):
        from safetensors.torch import load_file
        n = self.notebook
        _, stopped = self.run_training(self.root / "resumed", seconds=1e-12)
        self.assertFalse(stopped["complete"])
        self.assertEqual(stopped["step"], 1)
        _, resumed = self.run_training(self.root / "resumed", resume=True)
        _, full = self.run_training(self.root / "full")
        self.assertTrue(resumed["complete"])
        self.assertEqual(resumed["step"], full["step"])
        a = load_file(str(n.checkpoint_paths(self.root / "resumed")[-1] / "model.safetensors"))
        b = load_file(str(n.checkpoint_paths(self.root / "full")[-1] / "model.safetensors"))
        for name in a:
            n.torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        changed = replace(n.config, learning_rate=n.config.learning_rate * 2)
        with self.assertRaisesRegex(n.CorpusError, "settings changed"):
            self.run_training(self.root / "resumed", config=changed, resume=True)
        n.datasets["train"].books[0][1] += 1
        with self.assertRaisesRegex(n.CorpusError, "settings changed"):
            self.run_training(self.root / "resumed", resume=True)

    def test_final_partial_accumulation_covers_all_targets(self):
        n = self.notebook
        config = replace(n.config, epochs=1, max_steps=0, batch_size=3, gradient_accumulation=3, sample_every=0)
        _, result = self.run_training(self.root / "epoch", config=config)
        expected = sum(len(book) - 1 for book in n.datasets["train"].books)
        self.assertTrue(result["complete"])
        self.assertEqual(result["tokens_seen"], expected)

    def test_gradient_checkpointing_trains_and_exports_for_generation(self):
        n = self.notebook
        config = replace(n.config, gradient_checkpointing=True, max_steps=1)
        model, result = self.run_training(self.root / "checkpointing", config=config)
        self.assertTrue(result["complete"])
        self.assertFalse(model.is_gradient_checkpointing)
        saved = json.loads((self.root / "checkpointing" / "best_model" / "config.json").read_text())
        self.assertTrue(saved["use_cache"])
        self.assertTrue((self.root / "checkpointing" / "samples.jsonl").exists())

    def test_real_default_sampling_at_optimizer_step_100(self):
        n = self.notebook
        config = replace(n.config, epochs=100, max_steps=101, sample_every=100,
                         eval_every=200, save_every=100, log_every=100)
        _, result = self.run_training(self.root / "cadence", config=config)
        self.assertEqual(result["step"], 101)
        records = [json.loads(line) for line in (self.root / "cadence" / "samples.jsonl").read_text().splitlines()]
        self.assertEqual([record["step"] for record in records], [100])

    def test_samples_do_not_change_training_weights(self):
        from safetensors.torch import load_file
        n = self.notebook
        self.run_training(self.root / "samples")
        self.run_training(self.root / "silent", config=replace(n.config, sample_every=0))
        a = load_file(str(n.checkpoint_paths(self.root / "samples")[-1] / "model.safetensors"))
        b = load_file(str(n.checkpoint_paths(self.root / "silent")[-1] / "model.safetensors"))
        for name in a:
            n.torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_training_and_generation_cells_execute_directly(self):
        n = self.notebook
        with contextlib.redirect_stdout(io.StringIO()), patch.object(subprocess, "run", side_effect=AssertionError("No subprocess training")):
            exec(CELLS[6], n.__dict__)
            exec(CELLS[7], n.__dict__)
        self.assertTrue(n.result["complete"])
        self.assertTrue((n.OUTPUT_DIR / "best_model" / "model.safetensors").is_file())
        self.assertFalse((n.OUTPUT_DIR / "raw").exists())
        loaded = n.GPT2LMHeadModel.from_pretrained(n.OUTPUT_DIR / "best_model", local_files_only=True)
        self.assertEqual(loaded.config.vocab_size, len(n.tokenizer))


    def test_epoch_offsets_keep_all_targets_and_are_repeatable(self):
        n = self.notebook
        data = n.TokenWindows(['Mira ' * 50, 'Evening ' * 30], n.tokenizer, 17)
        layouts = []
        for epoch in (0, 1, 0):
            data.set_epoch(42, epoch)
            layouts.append(list(data.first_targets))
            offset = 0
            for book, end in zip(data.books, data.ends):
                self.assertEqual([token for index in range(offset, end) for token in data[index][1:]], book[1:].tolist())
                offset = end
        self.assertEqual(layouts[0], layouts[2])
        self.assertNotEqual(layouts[0], layouts[1])

    def test_sliding_evaluation_counts_each_target_once(self):
        n = self.notebook
        data = n.TokenWindows(['Mira ' * 50], n.tokenizer, 17)
        windows = list(n.evaluation_windows(data, 8))
        self.assertEqual([token for tokens, start in windows for token in tokens[start:]], data.books[0][1:].tolist())
        self.assertGreater(windows[1][1], 1)
        model = n.make_model(n.config, n.tokenizer)
        metrics = n.evaluate(model, data, replace(n.config, eval_stride=8), n.tokenizer, n.torch.device('cpu'))
        self.assertEqual(metrics['target_tokens'], len(data.books[0]) - 1)
        self.assertGreater(metrics['bits_per_byte'], 0)

    def test_early_stopping_uses_full_epochs_and_survives_resume(self):
        n = self.notebook
        config = replace(n.config, epochs=10, max_steps=0, early_stopping_patience=2,
                         sample_every=0, eval_every=100000)
        metrics = {'loss': 2.0, 'perplexity': 7.4, 'target_tokens': 100, 'bits_per_byte': 1.2}
        with patch.object(n, 'evaluate', return_value=metrics):
            _, result = self.run_training(self.root / 'stopping', config=config)
            self.assertTrue(result['complete'])
            self.assertEqual(result['stop_reason'], 'early_stopping')
            _, resumed = self.run_training(self.root / 'stopping', config=config, resume=True)
        self.assertEqual(result['step'], resumed['step'])
        self.assertEqual(resumed['stop_reason'], 'early_stopping')

    def test_resume_across_epoch_offsets_matches_uninterrupted(self):
        from safetensors.torch import load_file
        n = self.notebook
        config = replace(n.config, epochs=2, max_steps=0, sample_every=0)
        n.datasets['train'].set_epoch(config.seed, 0)
        batches = n.math.ceil(len(n.datasets['train']) / config.batch_size)
        steps = n.math.ceil(batches / config.gradient_accumulation)
        ticks = iter(range(100000))
        with patch.object(n, 'time', types.SimpleNamespace(monotonic=lambda: next(ticks))):
            _, stopped = self.run_training(self.root / 'boundary', config=config, seconds=steps + 1)
        self.assertEqual(stopped['step'], steps + 1)
        self.run_training(self.root / 'boundary', config=config, resume=True)
        self.run_training(self.root / 'uninterrupted', config=config)
        a = load_file(str(n.checkpoint_paths(self.root / 'boundary')[-1] / 'model.safetensors'))
        b = load_file(str(n.checkpoint_paths(self.root / 'uninterrupted')[-1] / 'model.safetensors'))
        for name in a:
            n.torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_invalid_optimization_settings_are_rejected(self):
        n = self.notebook
        for changes in ({'min_lr_ratio': float('nan')}, {'eval_stride': 0},
                        {'early_stopping_patience': -1}, {'early_stopping_min_delta': float('inf')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(n.config, **changes).validate()


if __name__ == "__main__":
    unittest.main()
