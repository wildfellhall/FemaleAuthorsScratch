"""Offline tests of the actual notebook functions; no source-file generation."""
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "main.py"
SOURCE = NOTEBOOK_PATH.read_text(encoding="utf-8")
HEADERS = list(re.finditer(r"^# %% Cell (\d+):[^\n]*\n", SOURCE, re.M))
CELLS = {int(header[1]): SOURCE[header.end():HEADERS[i + 1].start() if i + 1 < len(HEADERS) else len(SOURCE)]
         for i, header in enumerate(HEADERS)}
corpus = types.ModuleType("notebook_cleaner")
sys.modules[corpus.__name__] = corpus
exec(compile(CELLS[3], "cell-3", "exec"), corpus.__dict__)


class NotebookFormatTests(unittest.TestCase):
    def test_full_file_and_individual_cells_compile(self):
        self.assertEqual(list(CELLS), list(range(1, 8)))
        compile(SOURCE, str(NOTEBOOK_PATH), "exec")
        for number, source in CELLS.items():
            with self.subTest(cell=number):
                ast.parse(source, feature_version=(3, 10))
                compile(source, f"cell-{number}", "exec")

    def test_corpus_cell_is_directly_editable(self):
        scope = {}
        exec(CELLS[2], scope)
        self.assertTrue(corpus.load_books(scope["sources"]))
        self.assertIsInstance(scope["CLEANING_RULES"], dict)

    def test_no_drive_mount_or_generated_modules(self):
        tree = ast.parse(SOURCE)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertNotIn("google.colab", imports)
        self.assertNotIn("/content/drive", SOURCE)
        # The only subprocess is dependency installation, never the workflow itself.
        commands = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"]
        self.assertEqual(len(commands), 1)


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
    def test_growth_and_grouping_are_independent_of_list_length(self):
        entries = ["https://www.gutenberg.org/ebooks/123",
                   {"url": "https://www.gutenberg.org/ebooks/124", "work_id": "shared"}]
        before = corpus.load_books(entries)
        entries.append({"url": "https://www.gutenberg.org/ebooks/125", "work_id": "shared"})
        after = corpus.load_books(entries)
        self.assertEqual(before, after[:2])
        self.assertEqual(after[1].work_id, after[2].work_id)
        self.assertEqual(len(corpus.load_books(entries + [entries[0]])), 3)

    def test_conflicting_group_and_invalid_entries_rejected(self):
        url = "https://www.gutenberg.org/ebooks/123"
        for entries in ([{"url": url, "work_id": "a"}, {"url": url, "work_id": "b"}],
                        [], [123], [{"url": url, "work_id": 3}], ["http://example.com/123"]):
            with self.subTest(entries=entries), self.assertRaises(corpus.CorpusError):
                corpus.load_books(entries)

    def test_offline_preparation_reviews_failures_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            entries = [f"https://www.gutenberg.org/files/{i}/{i}-0.txt" for i in range(123, 127)]
            for book_id in ("123", "124", "125"):
                content = book_bytes(PARAGRAPH * 3 if book_id == "125" else NARRATIVE, book_id)
                (raw / f"{book_id}.txt").write_bytes(content)
                corpus.write_json(raw / f"{book_id}.json", {"id": book_id,
                    "source_url": f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                    "sha256": corpus.sha256(content)})
            with patch.object(corpus, "urlopen", side_effect=AssertionError("No network")), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(corpus.CorpusError, "audit.json"):
                    corpus.prepare_books(entries, {}, root, offline=True)
                documents = corpus.prepare_books(entries, {}, root, allow_partial=True, offline=True)
            self.assertEqual([row["status"] for row in json.loads((root / "audit.json").read_text())],
                             ["accepted", "duplicate", "needs_review", "failed"])
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["text"], corpus.clean_book(book_bytes(), {})[0])


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
