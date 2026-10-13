"""Shared, data-only corpus registry; the editable source stays in corpus-list.py."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import tokenize
import unicodedata
from urllib.parse import urlparse

DEFAULT_SOURCES = Path(__file__).with_name("corpus-list.py")


class CorpusError(Exception):
    """Invalid corpus input, download, or unresolved quality issue."""


@dataclass(frozen=True)
class Book:
    id: str
    url: str
    note: str = ""
    work_id: str = ""


def gutenberg_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in {"gutenberg.org", "www.gutenberg.org"} or parsed.query or parsed.fragment:
        raise CorpusError(f"Expected an HTTPS Gutenberg URL: {url!r}")
    patterns = (r"/files/(\d+)/\1(?:-[08])?\.txt", r"/cache/epub/(\d+)/pg\1\.txt", r"/ebooks/(\d+)(?:\.txt\.utf-8)?")
    for pattern in patterns:
        match = re.fullmatch(pattern, parsed.path)
        if match:
            return match[1]
    raise CorpusError(f"Unrecognized Gutenberg URL: {url!r}")


def load_books(path: Path = DEFAULT_SOURCES) -> list[Book]:
    """Parse literals, never execute source code. No corpus-size/title assumptions.

    Entries can be URL strings with comments, or dictionaries containing url,
    note, and a stable work_id shared by related volumes/editions.
    """
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    values = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "sources" for t in targets):
            values.append(node.value)
    if len(values) != 1 or not isinstance(values[0], ast.List):
        raise CorpusError("Expected exactly one literal Python list named sources.")
    comments = {t.start[0]: t.string.lstrip("# ").strip() for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.COMMENT}
    books, seen = [], {}
    for element in values[0].elts:
        try:
            value = ast.literal_eval(element)
        except (ValueError, TypeError) as exc:
            raise CorpusError("Every sources entry must be a literal URL or dictionary.") from exc
        if isinstance(value, str):
            value = {"url": value}
        if not isinstance(value, dict) or set(value) - {"url", "note", "work_id"} or not isinstance(value.get("url"), str):
            raise CorpusError("Corpus dictionaries support only url, note, and work_id.")
        if any(not isinstance(v, str) or not v.strip() for v in value.values()):
            raise CorpusError("Corpus entry fields must be nonempty strings.")
        book = Book(gutenberg_id(value["url"]), value["url"], value.get("note", comments.get(element.end_lineno, "")), value.get("work_id", ""))
        if book.id in seen:
            if seen[book.id].work_id != book.work_id:
                raise CorpusError(f"Conflicting work_id values for duplicate ebook {book.id}.")
            continue
        books.append(book)
        seen[book.id] = book
    if not books:
        raise CorpusError("The sources list is empty.")
    return books


def work_key(book: Book, rules: dict) -> str:
    if rules.get("work_id"):
        return rules["work_id"]
    if book.work_id:
        return book.work_id
    # Existing comments are supported; explicit work_id is safer for new entries.
    title = re.split(r"\s+-\s+", book.note, maxsplit=1)[0]
    title = re.sub(r"(?i)\(?\b(?:vol(?:ume)?s?\.?)\s*(?:\d+|[ivx]+)(?:\s*(?:of|[-–])\s*(?:\d+|[ivx]+))?\)?", "", title)
    title = re.sub(r"(?i)\(of\s+\d+\)", "", title)
    title = re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", title).casefold()).strip()
    return title or f"gutenberg-{book.id}"


def split_for(work_id: str, seed: str, validation: float, test: float) -> str:
    value = int(hashlib.sha256(f"{seed}\0{work_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if value < test else "validation" if value < test + validation else "train"
