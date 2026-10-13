"""Offline integration checks: data isolation, target coverage, train/resume/generate."""
import contextlib
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_prepare_corpus import book_bytes, NARRATIVE, RUNTIME_DIR

import corpus_registry
import prepare_corpus as corpus
import training_data as data
from train_gpt2 import (PRE_SAMPLE_TRAINER_SHA256, TrainConfig, checkpoint_paths,
                        generate_samples, make_model, parse_args, resume_recipe_matches, train)

HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers", "numpy"))


def make_corpus(root):
    sources, rules = root / 'sources.py', root / 'rules.json'
    entries = []
    for offset, split in enumerate(data.SPLITS):
        key = next(f'invented-{i}' for i in range(10000) if corpus.split_for(f'invented-{i}', 'gutenberg-corpus-v1', .05, .05) == split)
        book_id = str(120 + offset)
        url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
        entries.append({'url': url, 'work_id': key})
        raw = root / 'corpus' / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        content = book_bytes(NARRATIVE.replace('Mira', ['Ariel', 'Vesper', 'Zelda'][offset]), book_id)
        (raw / f'{book_id}.txt').write_bytes(content)
        corpus.write_json(raw / f'{book_id}.json', {'id': book_id, 'source_url': url, 'sha256': corpus.sha256(content)})
    sources.write_text('sources = ' + repr(entries))
    corpus.write_json(rules, {'version': 1, 'books': {}})
    with contextlib.redirect_stdout(io.StringIO()):
        code = corpus.main(['--sources', str(sources), '--overrides', str(rules), '--output', str(root / 'corpus'), '--offline', '--chunk-words', '13'])
    assert code == 0
    return data.resolve_run(root / 'corpus')


class RegistryTests(unittest.TestCase):
    def test_growth_explicit_groups_and_url_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'list.py'
            entries = [{'url': 'https://www.gutenberg.org/ebooks/123', 'work_id': 'group'}, {'url': 'https://www.gutenberg.org/cache/epub/456/pg456.txt', 'work_id': 'group'}]
            path.write_text('sources = ' + repr(entries))
            before = corpus_registry.load_books(path)
            entries.append('https://www.gutenberg.org/files/789/789-8.txt')
            path.write_text('sources = ' + repr(entries))
            after = corpus_registry.load_books(path)
            self.assertEqual(after[:2], before)
            self.assertEqual(len(after), 3)
            self.assertEqual(corpus.work_key(before[0], {}), corpus.work_key(before[1], {}))

    def test_literal_underscores_and_plural_transcriber_notes(self):
        text, report = corpus.clean_book(book_bytes(NARRATIVE + '\n\n_literal emphasis_ and some_identifier.\n\n[Transcriber’s Notes: Details.]'), {})
        self.assertIn('literal emphasis and some_identifier.', text)
        self.assertNotIn('Details', text)
        self.assertFalse(report['problems'])


class CorpusFixture:

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = make_corpus(self.root)


class DataTests(CorpusFixture, unittest.TestCase):
    def test_portable_pointer_and_lossless_documents(self):
        shutil.copytree(self.root / 'corpus', self.root / 'moved')
        moved = data.resolve_run(self.root / 'moved')
        self.assertTrue(moved.is_relative_to(self.root / 'moved'))
        report = data.inspect_corpus(moved)
        self.assertTrue(all(value['books'] == 1 for value in report['counts'].values()))
        for split in data.SPLITS:
            document = list(data.iter_documents(moved, split))[0]
            self.assertTrue(document['text'].startswith('CHAPTER I.'))
            self.assertTrue(document['text'].endswith('voice.\n'))

    def test_missing_chunk_fails_before_training(self):
        path = self.run / 'train.jsonl'
        rows = path.read_text().splitlines()
        path.write_text('\n'.join(rows[:-1]) + '\n')
        with self.assertRaisesRegex(corpus.CorpusError, 'Missing/reordered/changed'):
            data.inspect_corpus(self.run)

    def test_work_leakage_rejected(self):
        path = self.run / 'manifest.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]['work_id'] = rows[0]['work_id']
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(corpus.CorpusError, 'leaks across splits'):
            data.inspect_corpus(self.run)

    def test_review_gate_and_empty_validation(self):
        report_path = self.run / 'report.json'
        report = json.loads(report_path.read_text())
        report['complete'] = False
        corpus.write_json(report_path, report)
        with self.assertRaisesRegex(corpus.CorpusError, 'review items'):
            data.inspect_corpus(self.run)
        data.inspect_corpus(self.run, allow_partial=True)
        manifest_path = self.run / 'manifest.jsonl'
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        rows = [row for row in rows if row['split'] != 'validation']
        manifest_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.run / 'validation.jsonl').write_text('')
        with self.assertRaisesRegex(corpus.CorpusError, 'nonempty train and validation'):
            data.inspect_corpus(self.run, allow_partial=True)


@unittest.skipUnless(HAS_TRAINING, 'Install the dependencies in COLAB_ALL_IN_ONE.py Cell 5 for ML integration tests')
class TrainingTests(CorpusFixture, unittest.TestCase):
    def config(self):
        return TrainConfig(vocab_size=300, context_length=32, n_layer=1, n_head=2, n_embd=16,
                           batch_size=2, gradient_accumulation=2, epochs=1, max_steps=4,
                           eval_every=2, save_every=2, log_every=1, dropout=0.1,
                           gradient_checkpointing=True, sample_every=2,
                           sample_max_new_tokens=4, sample_prompts=['A'])

    def arguments(self, name):
        return parse_args(['--corpus', str(self.run), '--output', str(self.root / name),
                           '--work-dir', str(self.root / 'tokens'), '--device', 'cpu', '--max-seconds', '0'])

    def test_every_target_once_and_padding_not_eos(self):
        import numpy as np
        tokenizer = data.train_tokenizer(self.run, self.root / 'tok', 300, 2)
        with patch.object(data, 'iter_documents', wraps=data.iter_documents) as iterator:
            data.train_tokenizer(self.run, self.root / 'tok2', 300, 2)
            self.assertEqual([call.args[1] for call in iterator.call_args_list], ['train'])
        stats = data.tokenize_corpus(self.run, self.root / 'tokens', tokenizer)
        dataset = data.TokenWindows(self.root / 'tokens', 'train', 32)
        raw = np.fromfile(self.root / 'tokens' / 'train.bin', dtype='<u4').tolist()
        targets = [token for index in range(len(dataset)) for token in dataset[index][1:]]
        self.assertEqual(targets, raw[1:])
        self.assertEqual(len(targets), stats['train']['target_tokens'])
        self.assertEqual(targets[-1], tokenizer.eos_token_id)
        batch = data.collate_windows([dataset[0], dataset[len(dataset) - 1]], tokenizer.pad_token_id)
        self.assertTrue((batch['labels'][batch['attention_mask'] == 0] == -100).all())
        self.assertTrue((batch['labels'] == tokenizer.eos_token_id).any())
        self.assertNotEqual(tokenizer.eos_token_id, tokenizer.pad_token_id)
        self.assertEqual(tokenizer.decode(tokenizer.encode('Café — 雨', add_special_tokens=False)), 'Café — 雨')

    def test_training_resume_matches_uninterrupted_and_generation(self):
        import torch
        from safetensors.torch import load_file
        cfg = self.config()
        with contextlib.redirect_stdout(io.StringIO()):
            full = train(self.arguments('full'), replace(cfg, sample_every=0))
            args = self.arguments('resumed')
            args.max_seconds = 1e-9  # stop at a safe optimizer boundary
            partial = train(args, cfg)
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['step'], 1)
            self.assertNotIn('test', partial)
            args.resume, args.max_seconds = True, 0
            completed = train(args, cfg)
        self.assertTrue(completed['complete'])
        samples = [json.loads(line) for line in (self.root / 'resumed' / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in samples], [2, 4])
        self.assertFalse((self.root / 'full' / 'samples.jsonl').exists())
        self.assertEqual(full['tokens_seen'], completed['tokens_seen'])
        a = load_file(str(self.root / 'full' / 'checkpoint-4' / 'model.safetensors'))
        b = load_file(str(self.root / 'resumed' / 'checkpoint-4' / 'model.safetensors'))
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertEqual(completed['test']['target_tokens'], json.loads(next((self.root / 'tokens').glob('*/index.json')).read_text())['test']['target_tokens'])
        result = subprocess.run([sys.executable, str(RUNTIME_DIR / 'generate.py'), '--model', str(self.root / 'resumed' / 'best_model'), '--prompt', 'A', '--max-new-tokens', '5'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(corpus.CorpusError, 'settings'):
            train(args, replace(cfg, learning_rate=.01))

    def test_samples_preserve_rng_mode_cache_and_repeat_with_fixed_seed(self):
        import torch
        cfg = self.config()
        tokenizer = data.train_tokenizer(self.run, self.root / 'tok', 300, 2)
        model = make_model(cfg, tokenizer)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        device = torch.device('cpu')
        model.train()
        before = torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            first = generate_samples(model, tokenizer, cfg, device, 100, self.root)
            second = generate_samples(model, tokenizer, cfg, device, 200, self.root)
        self.assertEqual(first['samples'], second['samples'])
        self.assertIn('Samples at optimizer step 100', output.getvalue())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.generation_config.use_cache)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        model.eval()
        with contextlib.redirect_stdout(io.StringIO()):
            generate_samples(model, tokenizer, cfg, device, 300, self.root)
        self.assertFalse(model.training)

        def fail(**kwargs):
            torch.rand(10)
            raise RuntimeError('deliberate failure')
        model.train()
        with patch.object(model, 'generate', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'deliberate'):
            generate_samples(model, tokenizer, cfg, device, 400, self.root)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_samples_shorten_long_prompts_to_fit_context(self):
        import torch
        cfg = replace(self.config(), sample_prompts=['A very long prompt. ' * 100], sample_max_new_tokens=5)
        tokenizer = data.train_tokenizer(self.run, self.root / 'tok', 300, 2)
        model = make_model(cfg, tokenizer)
        with patch.object(model, 'generate', wraps=model.generate) as generate, contextlib.redirect_stdout(io.StringIO()):
            record = generate_samples(model, tokenizer, cfg, torch.device('cpu'), 100, self.root)
        call = generate.call_args.kwargs
        self.assertLessEqual(call['input_ids'].shape[1] + call['max_new_tokens'], cfg.context_length)
        self.assertTrue(call['use_cache'])
        self.assertTrue(record['samples'][0]['prompt_truncated'])
        self.assertLessEqual(record['samples'][0]['generated_tokens'], 5)

    def test_default_sample_cadence_is_100_optimizer_updates(self):
        self.assertEqual(TrainConfig().sample_every, 100)
        cfg = replace(self.config(), context_length=8, n_embd=8, batch_size=1,
                      gradient_accumulation=2, epochs=10, max_steps=101,
                      sample_every=100, sample_max_new_tokens=3,
                      eval_every=1000, save_every=1000, log_every=1000)
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.arguments('cadence'), cfg)
        samples = [json.loads(line) for line in (self.root / 'cadence' / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in samples], [100])

    def test_sample_settings_can_change_on_resume(self):
        cfg = self.config()
        args = self.arguments('settings')
        args.max_seconds = 1e-9
        with contextlib.redirect_stdout(io.StringIO()):
            train(args, cfg)
            args.resume, args.max_seconds = True, 0
            train(args, replace(cfg, sample_prompts=['She'], sample_every=1))
        samples = [json.loads(line) for line in (args.output / 'samples.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in samples], [2, 3, 4])
        self.assertEqual(samples[0]['samples'][0]['prompt'], 'She')

    def test_known_legacy_trainer_resumes_but_other_changes_do_not(self):
        import copy
        current = {'config': {'seed': 42}, 'corpus': 'unchanged', 'implementation': {
            'train_gpt2.py': 'current', 'training_data.py': 'same-data-code'}}
        old = copy.deepcopy(current)
        old['implementation']['train_gpt2.py'] = PRE_SAMPLE_TRAINER_SHA256
        self.assertTrue(resume_recipe_matches(old, current))
        for location, name, value in [('config', 'seed', 3), ('implementation', 'training_data.py', 'changed'), ('implementation', 'train_gpt2.py', 'unknown-code')]:
            altered = copy.deepcopy(old)
            altered[location][name] = value
            self.assertFalse(resume_recipe_matches(altered, current))
        old['corpus'] = 'different-corpus'
        self.assertFalse(resume_recipe_matches(old, current))

    def test_complete_epoch_includes_final_accumulation(self):
        cfg = replace(self.config(), max_steps=0, batch_size=3, gradient_accumulation=3)
        with contextlib.redirect_stdout(io.StringIO()):
            result = train(self.arguments('epoch'), cfg)
        index = json.loads(next((self.root / 'tokens').glob('*/index.json')).read_text())
        self.assertEqual(result['tokens_seen'], index['train']['target_tokens'])
        self.assertTrue(result['complete'])

    def test_windows_never_cross_books(self):
        import numpy as np
        directory = self.root / 'manual-tokens'
        directory.mkdir()
        first, second = [0, 11, 12, 13, 14, 15, 0], [0, 21, 22, 0]
        np.array(first + second, dtype='<u4').tofile(directory / 'train.bin')
        corpus.write_json(directory / 'index.json', {'train': {'tokens': len(first + second), 'books': [
            {'offset': 0, 'length': len(first)}, {'offset': len(first), 'length': len(second)}]}})
        windows = data.TokenWindows(directory, 'train', 4)
        self.assertEqual([windows[i] for i in range(len(windows))], [first[:4], first[3:], second])

    def test_incomplete_checkpoint_ignored(self):
        output = self.root / 'checkpoints'
        (output / 'checkpoint-100').mkdir(parents=True)
        (output / '.checkpoint-101.incomplete').mkdir()
        self.assertEqual(checkpoint_paths(output), [])
        corpus.write_json(output / 'checkpoint-100' / 'complete.json', {'step': 100})
        self.assertEqual(checkpoint_paths(output), [output / 'checkpoint-100'])

    def test_nonfinite_config_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.config(), learning_rate=float('nan')).validate()
        for changes in ({'sample_every': -1}, {'sample_every': True}, {'sample_prompts': []}, {'sample_prompts': ['']}, {'sample_max_new_tokens': 0}, {'sample_temperature': float('nan')}, {'sample_top_p': 1.1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config(), **changes).validate()


if __name__ == '__main__':
    unittest.main()
