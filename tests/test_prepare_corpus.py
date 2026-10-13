"""Small invented fixtures only. These tests never access the network."""
import ast
import atexit
import contextlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

# Test the only application source in the repository. Materialize its embedded
# modules into an isolated temporary directory, never into the checkout.
COLAB_PATH = Path(__file__).resolve().parents[1] / "COLAB_ALL_IN_ONE.py"
COLAB_SOURCE = COLAB_PATH.read_text(encoding="utf-8")
COLAB_TREE = ast.parse(COLAB_SOURCE, filename=str(COLAB_PATH))
EMBEDDED_FILES = {}
for statement in COLAB_TREE.body:
    call = statement.value if isinstance(statement, ast.Expr) else None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "write_text":
        continue
    destination = call.func.value
    if not isinstance(destination, ast.BinOp) or not isinstance(destination.op, ast.Div) or not isinstance(destination.left, ast.Name) or destination.left.id != "PROJECT":
        continue
    filename = ast.literal_eval(destination.right)
    if Path(filename).name != filename or filename in EMBEDDED_FILES:
        raise ValueError(f"Invalid/repeated embedded filename: {filename}")
    EMBEDDED_FILES[filename] = ast.literal_eval(call.args[0])

CELL_HEADERS = list(re.finditer(r"^# ={76}\n# CELL (\d+): [^\n]+\n# ={76}\n\n", COLAB_SOURCE, re.M))
COLAB_CELLS = {
    int(match[1]): COLAB_SOURCE[match.end():CELL_HEADERS[index + 1].start() if index + 1 < len(CELL_HEADERS) else len(COLAB_SOURCE)]
    for index, match in enumerate(CELL_HEADERS)
}
_RUNTIME = tempfile.TemporaryDirectory(prefix="femaleauthors-tests-")
atexit.register(_RUNTIME.cleanup)
RUNTIME_DIR = Path(_RUNTIME.name)
for filename, contents in EMBEDDED_FILES.items():
    (RUNTIME_DIR / filename).write_text(contents, encoding="utf-8")
sys.path.insert(0, str(RUNTIME_DIR))

import prepare_corpus as corpus

PARAGRAPH = (
    'Mira opened the door and watched the rain cross the empty garden. '
    'The old pear tree stood beside the gate, its leaves bright beneath the clouds. '
    'She remembered walking there with her sister every summer, carrying a little basket '
    'and arguing about which path would lead them home before sunset. '
    'Now the house was quiet, and she waited by the window for a familiar voice. '
)
NARRATIVE = 'CHAPTER I.\n\n' + PARAGRAPH + '\n\n' + PARAGRAPH + '\n\nCHAPTER II.\n\n' + PARAGRAPH


def book_bytes(body=NARRATIVE, book_id='123', newline='\n'):
    text = (
        'The Project Gutenberg eBook of An Invented Book\n\nTitle: An Invented Book\n'
        'Author: Invented Author\nLanguage: English\n\n'
        f'*** START OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n\n'
        + body + f'\n\n*** END OF THE PROJECT GUTENBERG EBOOK {book_id} ***\n'
        + 'Project Gutenberg license and donation information.\n'
    )
    return text.replace('\n', newline).encode('utf-8')


def rules_for(data, **kwargs):
    return {'raw_sha256': corpus.sha256(data), **kwargs}


class ColabFormatTests(unittest.TestCase):
    def test_entire_file_each_cell_and_embedded_python_compile(self):
        self.assertEqual(list(COLAB_CELLS), list(range(1, 14)))
        compile(COLAB_SOURCE, str(COLAB_PATH), 'exec')
        for number, source in COLAB_CELLS.items():
            with self.subTest(cell=number):
                ast.parse(source, feature_version=(3, 10))
                compile(source, f'cell-{number}', 'exec')
        for name, source in EMBEDDED_FILES.items():
            if name.endswith('.py'):
                with self.subTest(module=name):
                    ast.parse(source, feature_version=(3, 10))
                    compile(source, name, 'exec')
            elif name.endswith('.json'):
                json.loads(source)

    def test_cells_materialize_all_required_files_without_other_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            scope = {'PROJECT': project, 'subprocess': subprocess, 'sys': sys}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as commands:
                for number in range(2, 11):
                    exec(compile(COLAB_CELLS[number], f'cell-{number}', 'exec'), scope)
            self.assertEqual({path.name for path in project.iterdir()}, set(EMBEDDED_FILES))
            for name, source in EMBEDDED_FILES.items():
                self.assertEqual((project / name).read_bytes(), source.encode('utf-8'))
            self.assertEqual(commands.call_count, 2)
            self.assertEqual(commands.call_args_list[0].args[0], [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(project / 'requirements-colab.txt')])
            self.assertEqual(commands.call_args_list[1].args[0], ['nvidia-smi'])

    def test_required_runtime_files_and_config_are_present(self):
        self.assertEqual(set(EMBEDDED_FILES), {
            'corpus-list.py', 'corpus_registry.py', 'prepare_corpus.py',
            'training_data.py', 'train_gpt2.py', 'generate.py',
            'cleaning_rules.json', 'training_config.json', 'requirements-colab.txt',
        })
        from train_gpt2 import TrainConfig
        TrainConfig(**json.loads(EMBEDDED_FILES['training_config.json'])).validate()
        self.assertTrue(corpus.load_books(RUNTIME_DIR / 'corpus-list.py'))
        self.assertIsInstance(corpus.load_rules(RUNTIME_DIR / 'cleaning_rules.json'), dict)
        self.assertEqual(Path(corpus.__file__).parent, RUNTIME_DIR)

    def test_preparation_training_and_generation_cells_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / 'corpus' / 'runs' / 'test'
            prepared.mkdir(parents=True)
            output = root / 'run'
            output.mkdir()
            corpus.write_json(root / 'corpus' / 'latest.json', {'run': str(prepared), 'relative_run': 'runs/test'})
            corpus.write_json(prepared / 'report.json', {
                'sources_sha256': corpus.sha256(EMBEDDED_FILES['corpus-list.py'].encode()),
                'rules_sha256': corpus.sha256(EMBEDDED_FILES['cleaning_rules.json'].encode()),
                'counts': {'accepted': 1}, 'complete': True,
            })
            (prepared / 'manifest.jsonl').write_text('{"status":"accepted"}\n')
            corpus.write_json(output / 'run.json', {})
            corpus.write_json(output / 'result.json', {'complete': True})
            scope = {'PROJECT': RUNTIME_DIR, 'CORPUS': root / 'corpus', 'OUTPUT': output,
                     'REBUILD_CORPUS': False, 'ALLOW_PARTIAL_CORPUS': False,
                     'MAX_SESSION_SECONDS': 7200, 'Path': Path, 'json': json,
                     'subprocess': subprocess, 'sys': sys}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as commands:
                for number in (11, 12, 13):
                    exec(compile(COLAB_CELLS[number], f'cell-{number}', 'exec'), scope)
            self.assertEqual(scope['prepared'], prepared)
            self.assertEqual(commands.call_count, 2)
            training = commands.call_args_list[0].args[0]
            self.assertEqual(training[:2], [sys.executable, str(RUNTIME_DIR / 'train_gpt2.py')])
            self.assertIn('--resume', training)
            self.assertEqual(commands.call_args_list[1].args[0][:2], [sys.executable, str(RUNTIME_DIR / 'generate.py')])


class CleaningTests(unittest.TestCase):
    def test_wrappers_and_metadata(self):
        text, report = corpus.clean_book(book_bytes(), {})
        self.assertTrue(text.startswith('CHAPTER I.'))
        self.assertNotIn('Gutenberg', text)
        self.assertEqual(report['metadata']['author'], 'Invented Author')
        self.assertEqual(report['problems'], [])

    def test_toc_is_not_first_chapter(self):
        front = 'TITLE\nPublisher\n\nCONTENTS\n\nCHAPTER I.\nThe door\nCHAPTER II.\nThe garden\n\n'
        text, report = corpus.clean_book(book_bytes(front + NARRATIVE), {})
        self.assertEqual(text.count('CHAPTER I.'), 1)
        self.assertNotIn('CONTENTS', text)
        self.assertNotIn('TITLE', text)
        self.assertEqual(report['problems'], [])

    def test_prologue_preserved(self):
        body = 'TITLE\n\nPROLOGUE\n\n' + PARAGRAPH + '\n\n' + NARRATIVE
        text, _ = corpus.clean_book(book_bytes(body), {})
        self.assertTrue(text.startswith('PROLOGUE'))

    def test_letters_before_chapter_not_silently_lost(self):
        data = book_bytes('LETTER I\n\n' + PARAGRAPH + '\n\n' + NARRATIVE)
        _, report = corpus.clean_book(data, {})
        self.assertIn('possible_narrative_before_first_chapter', report['problems'])

    def test_unstructured_story_requires_boundary(self):
        data = book_bytes(PARAGRAPH * 3)
        _, report = corpus.clean_book(data, {})
        self.assertIn('narrative_start_not_identified', report['problems'])
        text, report = corpus.clean_book(data, rules_for(data, start={'text': 'Mira opened', 'occurrence': 1}))
        self.assertTrue(text.startswith('Mira opened'))
        self.assertEqual(report['problems'], [])

    def test_ambiguous_anchor_rejected(self):
        data = book_bytes()
        with self.assertRaises(corpus.CorpusError):
            corpus.clean_book(data, rules_for(data, start={'text': 'Mira opened'}))

    def test_rules_cannot_silently_drift(self):
        with self.assertRaisesRegex(corpus.CorpusError, 'raw_sha256'):
            corpus.clean_book(book_bytes(), {'start': {'text': 'CHAPTER I.'}})
        with self.assertRaisesRegex(corpus.CorpusError, 'raw_sha256'):
            corpus.clean_book(book_bytes(), {'raw_sha256': 'wrong'})

    def test_explicit_end_and_removal(self):
        data = book_bytes(NARRATIVE.replace('Mira opened', 'Mira[1] opened', 1) + '\n\nCATALOGUE\nPublisher notices.')
        rules = rules_for(data, start={'text': 'CHAPTER I.'}, end={'text': 'CATALOGUE'}, remove=[{'text': '[1]'}])
        text, report = corpus.clean_book(data, rules)
        self.assertNotIn('[1]', text)
        self.assertNotIn('CATALOGUE', text)
        self.assertEqual(report['problems'], [])

    def test_removal_mismatch_rejected(self):
        data = book_bytes()
        with self.assertRaises(corpus.CorpusError):
            corpus.clean_book(data, rules_for(data, remove=[{'text': 'nonexistent'}]))

    def test_multiline_captions_and_transcriber_notes(self):
        body = NARRATIVE + '\n\n[Illustration:\n\nA garden in the rain.]\n\nTRANSCRIBER’S NOTES\n\nSpelling was preserved.'
        text, report = corpus.clean_book(book_bytes(body), {})
        self.assertNotIn('Illustration', text)
        self.assertNotIn('Spelling', text)
        self.assertEqual(report['problems'], [])

    def test_footnotes_are_removed_or_flagged(self):
        data = book_bytes(NARRATIVE + '\n\n[Footnote 1: An explanatory note.]')
        text, report = corpus.clean_book(data, {})
        self.assertNotIn('explanatory', text)
        self.assertEqual(report['problems'], [])
        _, report = corpus.clean_book(book_bytes(NARRATIVE + '\n\n[1] An unlabelled note.'), {})
        self.assertIn('possible_footnote', report['problems'])

    def test_keep_book_matter(self):
        data = book_bytes('DEDICATION\nTo my sister.\n\n' + NARRATIVE + '\n\n[Footnote 1: A note.]')
        text, report = corpus.clean_book(data, {}, keep_book_matter=True)
        self.assertIn('DEDICATION', text)
        self.assertIn('Footnote', text)
        self.assertNotIn('Project Gutenberg', text)
        self.assertEqual(report['problems'], [])

    def test_verse_and_accents_preserved(self):
        verse = 'A quiet room\nA distant light\nA patient moon\nAcross the night'
        data = book_bytes(NARRATIVE + '\n\n' + verse + '\n\n“Café”—she said. _Italic words_.')
        text, _ = corpus.clean_book(data, {})
        self.assertIn(verse, text)
        self.assertIn('“Café”—she said. Italic words.', text)

    def test_prose_reflow_and_hyphens(self):
        wrapped = 'Here is a rather long invented line of prose ending in a well-\nknown expression that must retain its hyphen when reflowed.'
        text = corpus.tidy_layout(wrapped, [], True)
        self.assertIn('well-known', text)  # Keep the printed hyphen; do not invent a space.
        self.assertNotIn('\n', text.rstrip('\n'))
        self.assertIn('\nknown', corpus.tidy_layout(wrapped, [], False))

    def test_newlines_and_bom(self):
        text, report = corpus.clean_book(b'\xef\xbb\xbf' + book_bytes(newline='\r\n'), {})
        self.assertNotIn('\r', text)
        self.assertNotIn('\ufeff', text)
        self.assertEqual(report['problems'], [])

    def test_missing_reversed_and_duplicate_markers(self):
        for data in (b'No markers', book_bytes().replace(b'*** END', b'NO END'), book_bytes() + book_bytes()):
            with self.subTest(data=data[:50]):
                with self.assertRaises(corpus.CorpusError):
                    corpus.clean_book(data, {})
        text = '*** END OF THE PROJECT GUTENBERG EBOOK 123 ***\n*** START OF THE PROJECT GUTENBERG EBOOK 123 ***'
        with self.assertRaises(corpus.CorpusError):
            corpus.clean_book(text.encode(), {})

    def test_html_rejected(self):
        with self.assertRaises(corpus.CorpusError):
            corpus.clean_book(b'<html><body>Access denied</body></html>', {})

    def test_residual_boilerplate_cannot_be_waived(self):
        data = book_bytes(NARRATIVE + '\n\nProject Gutenberg license')
        _, report = corpus.clean_book(data, rules_for(data, accept_findings=['gutenberg_boilerplate']))
        self.assertIn('gutenberg_boilerplate', report['problems'])

    def test_damaged_encoding_quarantined(self):
        _, report = corpus.clean_book(book_bytes(NARRATIVE + '\n\nBad \ufffd letter.\x00'), {})
        self.assertIn('replacement_character', report['problems'])
        self.assertIn('control_character', report['problems'])
        data = book_bytes(NARRATIVE + '\n\nCafé.').decode().encode('cp1252')
        text, report = corpus.clean_book(data, {})
        self.assertIn('Café', text)
        self.assertIn('legacy_encoding_needs_review', report['problems'])


class PipelineTests(unittest.TestCase):
    def test_source_is_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sources.py'
            path.write_text('raise RuntimeError("must never execute")\nsources = [\n"https://www.gutenberg.org/files/123/123-0.txt", # An Invented Book\n]\n')
            books = corpus.load_books(path)
            self.assertEqual(books[0].id, '123')
            self.assertEqual(books[0].note, 'An Invented Book')

    def test_nonliteral_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sources.py'
            path.write_text('sources = [str(123)]')
            with self.assertRaises(corpus.CorpusError):
                corpus.load_books(path)

    def test_chunks_are_lossless_and_bounded(self):
        for maximum in (1, 17, 1024):
            pieces = list(corpus.chunks(NARRATIVE, maximum))
            self.assertEqual(''.join(pieces), NARRATIVE)
            self.assertTrue(all(0 < len(piece.split()) <= maximum for piece in pieces))

    def test_volumes_share_split(self):
        def key(note):
            return corpus.work_key(corpus.Book('123', '', note), {})
        for a, b in [('Daisy Burns (Volume 1) by Julia Kavanagh', 'Daisy Burns (Volume 2) by Julia Kavanagh'), ('Lodore, Vol. 1 (of 3) by Shelley', 'Lodore, Vol. 3 (of 3) by Shelley'), ('Cecilia — Volume 1 by Burney', 'Cecilia — Volume 2 by Burney')]:
            self.assertEqual(key(a), key(b))
            self.assertEqual(corpus.split_for(key(a), 'seed', .05, .05), corpus.split_for(key(b), 'seed', .05, .05))

    def test_complete_offline_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, rules = root / 'sources.py', root / 'rules.json'
            ids = ['123', '124', '125', '126']
            sources.write_text('sources = ' + repr([f'https://www.gutenberg.org/files/{i}/{i}-0.txt' for i in ids]))
            corpus.write_json(rules, {'version': 1, 'books': {}})
            raw = root / 'out' / 'raw'
            raw.mkdir(parents=True)
            for book_id in ids[:3]:
                data = book_bytes(NARRATIVE if book_id != '125' else PARAGRAPH * 3, book_id)
                (raw / f'{book_id}.txt').write_bytes(data)
                corpus.write_json(raw / f'{book_id}.json', {'id': book_id, 'source_url': f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt', 'sha256': corpus.sha256(data)})
            arguments = ['--sources', str(sources), '--output', str(root / 'out'), '--overrides', str(rules), '--offline', '--chunk-words', '40']
            with patch.object(corpus, 'urlopen', side_effect=AssertionError('Tests cannot access network')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(corpus.main(arguments), 1)
            latest = json.loads((root / 'out' / 'latest.json').read_text())
            run = Path(latest['run'])
            records = [json.loads(line) for line in (run / 'manifest.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in records], ['accepted', 'duplicate', 'needs_review', 'failed'])
            samples = [json.loads(line) for split in ('train', 'validation', 'test') for line in (run / f'{split}.jsonl').read_text().splitlines()]
            self.assertTrue(samples)
            self.assertEqual({s['book_id'] for s in samples}, {'123'})
            self.assertEqual(''.join(s['text'] for s in samples), corpus.clean_book(book_bytes(), {})[0])
            self.assertFalse((run / 'clean').exists())
            self.assertFalse((run / 'review').exists())
            self.assertTrue((run / 'audit.html').exists())
            self.assertFalse(latest['complete'])
            # A second run has no stale samples from the first.
            with patch.object(corpus, 'urlopen', side_effect=AssertionError('Tests cannot access network')), contextlib.redirect_stdout(io.StringIO()):
                corpus.main(arguments + ['--ids', '125'])
            second = Path(json.loads((root / 'out' / 'latest.json').read_text())['run'])
            self.assertNotEqual(run, second)
            self.assertEqual((second / 'train.jsonl').read_text(), '')


class FakeResponse(io.BytesIO):
    def __init__(self, data, content_type='text/plain'):
        super().__init__(data)
        self.headers = {'Content-Type': content_type}
        self.url = 'https://example.test/book.txt'


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.book = corpus.Book('123', 'https://www.gutenberg.org/files/123/123-0.txt')
        self.downloader = corpus.Downloader(self.root, 'https://example.test', 0, 1, 0, False)

    def test_no_disk_download_by_default(self):
        with patch.object(corpus, 'urlopen', return_value=FakeResponse(book_bytes())):
            data, metadata = self.downloader.fetch(self.book)
        self.assertEqual(data, book_bytes())
        self.assertEqual(metadata['sha256'], corpus.sha256(data))
        self.assertFalse((self.root / 'raw').exists())

    def test_fallback_after_404(self):
        error = HTTPError('https://example.test', 404, 'Not found', {}, None)
        with patch.object(corpus, 'urlopen', side_effect=[error, FakeResponse(book_bytes())]) as request:
            self.downloader.fetch(self.book)
        self.assertEqual(request.call_count, 2)

    def test_access_denied_not_bypassed(self):
        error = HTTPError('https://example.test', 403, 'Forbidden', {}, None)
        with patch.object(corpus, 'urlopen', side_effect=error) as request:
            with self.assertRaises(corpus.CorpusError):
                self.downloader.fetch(self.book)
            with self.assertRaises(corpus.CorpusError):
                self.downloader.fetch(corpus.Book('456', 'https://www.gutenberg.org/files/456/456-0.txt'))
        self.assertEqual(request.call_count, 1)

    def test_wrong_book_id_rejected(self):
        with patch.object(corpus, 'urlopen', return_value=FakeResponse(book_bytes(book_id='999'))):
            with self.assertRaises(corpus.CorpusError):
                self.downloader.fetch(self.book)

    def test_html_is_not_cached(self):
        self.downloader.cache_raw = True
        with patch.object(corpus, 'urlopen', return_value=FakeResponse(b'<html>Error</html>', 'text/html')):
            with self.assertRaises(corpus.CorpusError):
                self.downloader.fetch(self.book)
        self.assertFalse((self.root / 'raw').exists())

    def test_cache_reuse_and_corruption(self):
        self.downloader.cache_raw = True
        with patch.object(corpus, 'urlopen', return_value=FakeResponse(book_bytes())):
            self.downloader.fetch(self.book)
        with patch.object(corpus, 'urlopen', side_effect=AssertionError('Cache should be used')):
            self.downloader.fetch(self.book)
            (self.root / 'raw' / '123.txt').write_bytes(b'corrupt')
            with self.assertRaises(corpus.CorpusError):
                self.downloader.fetch(self.book)

    def test_retry_after(self):
        self.assertEqual(corpus.retry_delay('61', 0), 61)
        self.assertEqual(corpus.retry_delay('', 2), 16)
        self.assertEqual(corpus.retry_delay('Thu, 01 Jan 1970 00:00:00 GMT', 0), 0)

    def test_network_retry(self):
        self.downloader.retries = 1
        with patch.object(corpus, 'time') as clock, patch.object(corpus, 'urlopen', side_effect=[URLError('temporary'), FakeResponse(book_bytes())]) as request:
            clock.monotonic.return_value = 100
            self.downloader.fetch(self.book)
        self.assertEqual(request.call_count, 2)
        clock.sleep.assert_any_call(2)


if __name__ == '__main__':
    unittest.main()
