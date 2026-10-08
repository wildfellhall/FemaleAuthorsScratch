"""Download, audit, and prepare a Gutenberg corpus. Python 3.10+, no dependencies.
    Trying to be extra careful with the parser since this is a from-scratch model. 
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import io
from http.client import HTTPException
import json
import math
from pathlib import Path
import re
import sys
import time
import tokenize
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime

VERSION = "1.0"
DEFAULT_MIRROR = "https://gutenberg.pglaf.org"
URL_RE = re.compile(r"https://(?:www\.)?gutenberg\.org/files/(\d+)/\1-0\.txt\Z")
START = re.compile(r"(?im)^[ \t]*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*\*{3}[^\n]*$")
END = re.compile(r"(?im)^[ \t]*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG (?:EBOOK|ETEXT)\b[^\n]*\*{3}[^\n]*$")
LEGACY_END = re.compile(r"(?im)^[ \t]*End of (?:the )?Project Gutenberg(?:'s)? (?:EBook|Etext)\b[^\n]*$")


class CorpusError(Exception):
    """A recoverable per-book input, network, or quality error."""


@dataclass(frozen=True)
class Book:
    id: str
    url: str
    note: str = ""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_books(path: Path) -> list[Book]:
    """Read the literal list without importing/executing the Python source."""
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
    books, seen = [], set()
    for element in values[0].elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise CorpusError("Every sources entry must be a literal URL string.")
        match = URL_RE.fullmatch(element.value)
        if not match:
            raise CorpusError(f"Unexpected Gutenberg URL: {element.value!r}")
        book_id = match[1]
        if book_id not in seen:
            books.append(Book(book_id, element.value, comments.get(element.end_lineno, "")))
            seen.add(book_id)
    if not books:
        raise CorpusError("The sources list is empty.")
    return books


def decode_book(data: bytes) -> tuple[str, str]:
    # Never use errors='ignore' or errors='replace': lost characters are training errors.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252"), "cp1252"
        except UnicodeDecodeError as exc:
            raise CorpusError(f"Cannot decode book without losing characters: {exc}") from exc


def normalize_source(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).lstrip("\ufeff")


def gutenberg_body(text: str) -> tuple[str, dict]:
    text = normalize_source(text)
    if re.search(r"(?is)<(?:!doctype\s+html|html|body)\b", text[:2048]):
        raise CorpusError("HTML response, not a plain-text book.")
    starts = list(START.finditer(text))
    ends = list(END.finditer(text)) or list(LEGACY_END.finditer(text))
    if len(starts) != 1 or len(ends) != 1 or ends[0].start() <= starts[0].end():
        raise CorpusError("Missing, ambiguous, or reversed Gutenberg start/end markers; refusing to guess.")
    start, end = starts[0], ends[0]
    header = text[:start.start()]
    metadata = {}
    for key in ("Title", "Author", "Language", "Release date", "Release Date"):
        m = re.search(rf"(?im)^{re.escape(key)}:\s*(.+)$", header)
        if m:
            metadata[key.lower().replace(" ", "_")] = m[1].strip()
    metadata.update(start_marker=start.group().strip(), end_marker=end.group().strip(), wrapper_removed_chars=len(text) - (end.start() - start.end()))
    return text[start.end():end.start()].strip(), metadata


class Downloader:
    def __init__(self, root: Path, mirror: str, delay: float, timeout: float, retries: int, offline: bool, cache_raw: bool = False):
        self.root, self.mirror = root, mirror.rstrip("/")
        self.delay, self.timeout, self.retries, self.offline = delay, timeout, retries, offline
        self.last_request = 0.0
        self.cache_raw = cache_raw
        self.blocked_reason = None

    def candidates(self, book: Book) -> list[str]:
        digits = "/".join(book.id[:-1]) if len(book.id) > 1 else "0"
        base = f"{self.mirror}/{digits}/{book.id}"
        return [f"{base}/{book.id}-0.txt", f"{self.mirror}/cache/epub/{book.id}/pg{book.id}.txt", f"{base}/{book.id}-8.txt", f"{base}/{book.id}.txt"]

    def fetch(self, book: Book) -> tuple[bytes, dict]:
        raw = self.root / "raw" / f"{book.id}.txt"
        meta_path = raw.with_suffix(".json")
        if raw.exists() and meta_path.exists():
            data, meta = raw.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sha256") != sha256(data):
                raise CorpusError(f"Cached download checksum mismatch: {raw}")
            if meta.get("id") != book.id or meta.get("source_url") != book.url:
                raise CorpusError(f"Cached download identity mismatch: {raw}")
            return data, meta
        if self.offline:
            raise CorpusError(f"No complete cached download for {book.id}; run without --offline.")
        if self.blocked_reason:
            raise CorpusError(self.blocked_reason)
        missing = []
        for url in self.candidates(book):
            for attempt in range(self.retries + 1):
                time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
                request = Request(url, headers={"User-Agent": f"GutenbergCorpusPrep/{VERSION} (sequential; cached; research corpus)", "Accept": "text/plain"})
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        data = response.read(30 * 1024 * 1024 + 1)
                        if len(data) > 30 * 1024 * 1024:
                            raise CorpusError("Response exceeds the 30 MiB per-book limit.")
                        if "html" in response.headers.get("Content-Type", "").lower():
                            raise CorpusError(f"Server returned HTML for {url}")
                        resolved_url = response.url
                    decoded, encoding = decode_book(data)
                    _, wrapper = gutenberg_body(decoded)
                    for marker in (wrapper["start_marker"], wrapper["end_marker"]):
                        mid = re.search(r"(?:EBOOK|ETEXT)\s+(\d+)\s*\*", marker, re.I)
                        if mid and mid[1] != book.id:
                            raise CorpusError(f"Downloaded ID {mid[1]} does not match requested {book.id}.")
                    meta = {"id": book.id, "source_url": book.url, "download_url": resolved_url, "sha256": sha256(data), "bytes": len(data), "encoding": encoding, "downloaded_at": datetime.now(timezone.utc).isoformat()}
                    if self.cache_raw:
                        raw.parent.mkdir(parents=True, exist_ok=True)
                        tmp = raw.with_suffix(".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(raw)
                        write_json(meta_path, meta)
                    return data, meta
                except HTTPError as exc:
                    if exc.code in (404, 410):
                        missing.append(f"{exc.code}: {url}")
                        break
                    if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                        retry_after = exc.headers.get("Retry-After", "")
                        pause = retry_delay(retry_after, attempt)
                        if pause > 300:
                            self.blocked_reason = "Server requested an extended pause; retry this run later."
                            raise CorpusError(f"Server requested a {pause:.0f}s pause; retry this book later.") from exc
                        time.sleep(pause)
                        continue
                    if exc.code in (401, 403, 429):
                        self.blocked_reason = f"Server refused access (HTTP {exc.code}); further network requests stopped."
                    raise CorpusError(f"HTTP {exc.code}: {url}") from exc
                except (URLError, TimeoutError, ConnectionError, HTTPException) as exc:
                    if attempt < self.retries:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise CorpusError(f"Download failed: {url}: {exc}") from exc
        raise CorpusError("No plain-text edition found. " + "; ".join(missing))


def retry_delay(value: str, attempt: int) -> float:
    if value.isdigit():
        return float(value)
    if value:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            pass
    return 2 ** (attempt + 2)


CHAPTER = re.compile(
    r"(?im)^[ \t]*(?:CHAPTER|CHAP\.)[ \t]+(?:[IVXLCDM]+|\d+|ONE|TWO|THREE|FIRST|SECOND|THIRD)"
    r"\b[^\n]{0,140}$"
)
PROLOGUE = re.compile(r"(?im)^[ \t]*(?:PROLOGUE|PRELUDE|INTRODUCTORY CHAPTER)[. \t]*$")
SUSPECT = {
    "gutenberg_boilerplate": r"(?i)project\s+gutenberg|gutenberg\.org|www\.pgdp\.net|distributed\s+proofread",
    "transcriber_material": r"(?i)transcrib(?:er|ers|er's|ers’|er’s).{0,15}(?:note|comment)|digitiz(?:ed|ation)|scann?ed\s+(?:by|from)",
    "contents_or_illustration_list": r"(?im)^\s*(?:(?:table of )?contents|list of illustrations|illustrations)[. \t]*$",
    "publisher_material": r"(?im)^\s*(?:all rights reserved|printed (?:by|in)|published by|copyright(?:ed)?\b|advertisements\b)",
    "possible_back_matter": r"(?im)^\s*(?:by the same author|other books by|books by|publisher['’]s (?:note|catalogue|catalog|announcement)|appendix|editor['’]s notes?)[.: \t]*$",
    "remaining_illustration": r"(?i)\[illustration\b|\[page\s+\d+\]|\[pg\.?\s*\d+\]",
    "possible_footnote": r"(?im)\[(?:\d{1,4}|[A-Z]|[*†‡])\]|^\s*(?:footnotes?|endnotes?)\b",
    "markup": r"(?i)</?(?:html|body|div|span|p|a|img)\b[^>]*>|&(?:amp|lt|gt|nbsp);",
    "replacement_character": "\ufffd",
    "possible_encoding_damage": r"Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[€\u0080-\u009f]",
    "control_character": r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]",
}


def record_removal(audit: list, reason: str, removed: str) -> None:
    if removed:
        audit.append({"reason": reason, "chars": len(removed), "sha256": sha256(removed.encode("utf-8")), "preview": removed[:200]})


def substitute(text: str, pattern: str, reason: str, audit: list, replacement: str = "") -> str:
    removed_hash = hashlib.sha256()
    total_chars, count, preview = 0, 0, ""
    def replace(match):
        nonlocal total_chars, count, preview
        removed = match.group()
        removed_hash.update(removed.encode("utf-8"))
        total_chars += len(removed)
        count += 1
        if len(preview) < 200:
            preview += removed[:200 - len(preview)]
        return replacement
    result = re.sub(pattern, replace, text)
    if count:
        audit.append({"reason": reason, "count": count, "chars": total_chars, "sha256": removed_hash.hexdigest(), "preview": preview})
    return result


def anchor_position(text: str, spec: dict) -> int:
    if not isinstance(spec, dict) or not isinstance(spec.get("text"), str) or not spec["text"]:
        raise CorpusError("An anchor requires a nonempty literal 'text' field.")
    positions = [m.start() for m in re.finditer(re.escape(normalize_source(spec["text"])), text)]
    occurrence = spec.get("occurrence")
    if occurrence is None:
        if len(positions) != 1:
            raise CorpusError(f"Anchor must match exactly once (found {len(positions)}): {spec['text'][:80]!r}")
        return positions[0]
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or not 1 <= occurrence <= len(positions):
        raise CorpusError(f"Invalid occurrence for anchor {spec['text'][:80]!r}")
    return positions[occurrence - 1]


def trim_to_narrative(text: str, rules: dict, audit: list) -> tuple[str, list[str], str]:
    problems = []
    method = "explicit"
    if "start" in rules:
        start = anchor_position(text, rules["start"])
    else:
        # TOC entries are followed by more short entries, not sustained prose.
        headings = list(CHAPTER.finditer(text))
        candidates = []
        for index, heading in enumerate(headings):
            next_pos = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            after = text[heading.end():min(next_pos, heading.end() + 5000)]
            paragraphs = re.split(r"\n[ \t]*\n", after.strip())[:5]
            if any(len(p.split()) >= 45 and re.search(r"[.!?][\"’”']?(?:\s|$)", p) for p in paragraphs):
                candidates.append(heading)
        if candidates:
            start = candidates[0].start()
            method = "chapter_heuristic"
            prologues = list(PROLOGUE.finditer(text[:start]))
            if prologues:
                start = prologues[-1].start()
            if start > len(text) * 0.25:
                problems.append("large_front_matter_cut")
            prefix = text[:start]
            if re.search(r"(?im)^\s*(?:LETTER\s+(?:I|1)|PREFATORY TALE)\b", prefix):
                problems.append("possible_narrative_before_first_chapter")
        else:
            start = 0
            method = "unresolved"
            problems.append("narrative_start_not_identified")
    end = anchor_position(text, rules["end"]) if "end" in rules else len(text)
    if end <= start:
        raise CorpusError("The narrative end anchor precedes its start anchor.")
    record_removal(audit, "front_matter", text[:start])
    record_removal(audit, "back_matter", text[end:])
    return text[start:end], problems, method


def tidy_layout(text: str, audit: list, reflow: bool) -> str:
    # NFC, not NFKC: preserve ligatures, accents, spelling, case, and punctuation.
    text = text.replace("\u00a0", " ").replace("\u00ad", "").replace("\ufeff", "")
    text = text.replace("\f", "\n\n")
    text = substitute(text, r"(?m)^[ \t]*(?:\[Page\s+\d+\]|\[Pg\.?\s*\d+\])[ \t]*$", "page_label", audit)
    # Underscores are Gutenberg's emphasis notation; keep the enclosed words.
    text = substitute(text, r"_", "emphasis_marker", audit)
    paragraphs = []
    for paragraph in re.split(r"\n[ \t]*\n+", text.strip()):
        lines = [line.rstrip() for line in paragraph.splitlines()]
        if not lines:
            continue
        # Reflow wrapped prose only. Short/indented lines may be verse or a letter.
        prose = len(lines) > 1 and all(len(line.strip()) >= 45 for line in lines[:-1])
        indented = any(line.startswith(("  ", "\t")) for line in lines[1:])
        if reflow and prose and not indented:
            joined = "\n".join(line.strip() for line in lines)
            # Keep printed hyphens, but avoid inserting a space into well-\nknown.
            joined = re.sub(r"(?<=\w)-\n(?=\w)", "-", joined)
            paragraphs.append(joined.replace("\n", " "))
        else:
            paragraphs.append("\n".join(lines).strip())
    return "\n\n".join(p for p in paragraphs if p).strip() + "\n"


def clean_book(data: bytes, rules: dict, keep_book_matter: bool = False, reflow: bool = True) -> tuple[str, dict]:
    decoded, encoding = decode_book(data)
    body, metadata = gutenberg_body(decoded)
    audit, problems = [], []
    if rules and rules.get("raw_sha256") != sha256(data):
        raise CorpusError("Cleaning rules require the matching raw_sha256; the edition may have changed.")
    original_chars = len(body)
    # Explicit digital artifacts can occur within either boundary.
    body = substitute(body, r"(?is)\[(?:Illustration\b|Illustrations\b)[^\[\]]{0,12000}\]", "illustration_caption", audit)
    body = substitute(body, r"(?is)\[Transcriber[’']?s?\s+Note\b[^\[\]]{0,12000}\]", "bracketed_transcriber_note", audit)
    # A terminal transcriber's section is safely bounded only near the end.
    notes = list(re.finditer(r"(?im)^[ \t]*(?:[*_ ]*)TRANSCRIBER[’']?S?[ \t]+NOTES?[.: \t*_]*$", body))
    if notes and notes[-1].start() > len(body) * 0.8:
        record_removal(audit, "terminal_transcriber_notes", body[notes[-1].start():])
        body = body[:notes[-1].start()]
    # Apply explicit anchors to the normalized, artifact-stripped text.
    if not keep_book_matter:
        body, boundary_problems, method = trim_to_narrative(body, rules, audit)
        problems.extend(boundary_problems)
    else:
        method = "keep_book_matter"
    for removal in rules.get("remove", []):
        if not isinstance(removal, dict) or not isinstance(removal.get("text"), str) or not removal["text"]:
            raise CorpusError("Each removal requires a nonempty literal 'text' field.")
        literal = normalize_source(removal["text"])
        expected = removal.get("count", 1)
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0 or body.count(literal) != expected:
            raise CorpusError(f"Removal count mismatch: {literal[:80]!r}")
        record_removal(audit, "explicit_removal", literal * expected)
        body = body.replace(literal, "")
    if not keep_book_matter:
        # Remove only explicitly labelled footnotes, including their own reference.
        # Bare [1]/[A] references are ambiguous and sent for review, never guessed.
        body = substitute(body, r"(?is)\[Footnote\b[^\[\]]{0,12000}\]", "labelled_footnote", audit)
    clean = tidy_layout(body, audit, reflow)
    findings = {}
    for name, pattern in SUSPECT.items():
        if keep_book_matter and name in {"contents_or_illustration_list", "publisher_material", "possible_footnote", "possible_back_matter"}:
            continue
        hits = list(re.finditer(pattern, clean))
        if hits:
            findings[name] = {"count": len(hits), "examples": [clean[max(0, m.start()-60):m.end()+100] for m in hits[:3]]}
            problems.append(name)
    if encoding != "utf-8":
        problems.append("legacy_encoding_needs_review")
    if len(clean.split()) < 100:
        problems.append("very_short_book")
    if not keep_book_matter and len(clean) < original_chars * 0.5:
        problems.append("more_than_half_removed")
    accepted = rules.get("accept_findings", [])
    if not isinstance(accepted, list) or not all(isinstance(x, str) for x in accepted):
        raise CorpusError("accept_findings must be a list of finding names.")
    unwaivable = {"gutenberg_boilerplate", "replacement_character", "control_character", "markup"}
    problems = sorted(set(problems) - (set(accepted) - unwaivable))
    return clean, {"encoding": encoding, "metadata": metadata, "boundary_method": method, "removals": audit, "findings": findings, "problems": problems, "words": len(clean.split()), "characters": len(clean), "original_body_characters": original_chars}


def work_key(book: Book, rules: dict) -> str:
    if rules.get("work_id"):
        return rules["work_id"]
    # Source comments are provenance, not book text. Volumes share a split.
    title = re.split(r"\s+-\s+", book.note, maxsplit=1)[0]
    title = re.sub(r"(?i)\(?\b(?:vol(?:ume)?s?\.?)\s*(?:\d+|[ivx]+)(?:\s*(?:of|[-–])\s*(?:\d+|[ivx]+))?\)?", "", title)
    title = re.sub(r"(?i)\(of\s+\d+\)", "", title)
    title = re.sub(r"[^\w]+", " ", unicodedata.normalize("NFC", title).casefold()).strip()
    return title or f"gutenberg-{book.id}"


def split_for(work_id: str, seed: str, validation: float, test: float) -> str:
    value = int(sha256(f"{seed}\0{work_id}".encode())[:16], 16) / 2**64
    if value < test:
        return "test"
    if value < test + validation:
        return "validation"
    return "train"


def chunks(text: str, max_words: int):
    """Nonoverlapping character slices. Concatenating chunks restores exact text."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    start, count = 0, 0
    for word in re.finditer(r"\S+", text):
        if count == max_words:
            yield text[start:word.start()]
            start, count = word.start(), 0
        count += 1
    if text[start:].strip():
        yield text[start:]


def load_rules(path: Path) -> dict:
    if not path.exists():
        raise CorpusError(f"Cleaning rules file not found: {path}")
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or rules.get("version") != 1 or not isinstance(rules.get("books"), dict):
        raise CorpusError('Rules must contain "version": 1 and a "books" object.')
    allowed = {"raw_sha256", "start", "end", "remove", "accept_findings", "work_id", "notes"}
    for book_id, rule in rules["books"].items():
        if not book_id.isdigit() or not isinstance(rule, dict) or set(rule) - allowed:
            raise CorpusError(f"Invalid or unknown rule fields for {book_id}.")
    return rules["books"]


def prepare(books: list[Book], downloader: Downloader, args) -> int:
    rules = load_rules(args.overrides)
    # A new directory per invocation prevents failed/partial runs reusing stale text.
    run = args.output / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True)
    records, seen_hashes, counts = [], {}, Counter()
    outputs = {}
    try:
        for split in ("train", "validation", "test"):
            outputs[split] = (run / f"{split}.jsonl").open("w", encoding="utf-8")
        with (run / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
            for index, book in enumerate(books, 1):
                record = {"id": book.id, "source_url": book.url, "source_note": book.note}
                try:
                    data, source = downloader.fetch(book)
                    record["download"] = source
                    clean, details = clean_book(data, rules.get(book.id, {}), args.keep_book_matter, not args.preserve_linebreaks)
                    record.update(details)
                    record["clean_sha256"] = sha256(clean.encode("utf-8"))
                    record["opening"], record["closing"] = clean[:800], clean[-800:]
                    if details["problems"]:
                        record["status"] = "needs_review"
                        if args.write_books:
                            target = run / "review" / f"{book.id}.txt"
                            target.parent.mkdir(exist_ok=True)
                            target.write_text(clean, encoding="utf-8")
                    elif record["clean_sha256"] in seen_hashes:
                        record.update(status="duplicate", duplicate_of=seen_hashes[record["clean_sha256"]])
                    else:
                        record["status"] = "accepted"
                        seen_hashes[record["clean_sha256"]] = book.id
                        work_id = work_key(book, rules.get(book.id, {}))
                        split = split_for(work_id, args.seed, args.validation_fraction, args.test_fraction)
                        record.update(work_id=work_id, split=split, samples=0)
                        for number, chunk in enumerate(chunks(clean, args.chunk_words)):
                            sample = {"id": f"{book.id}:{number}", "book_id": book.id, "work_id": work_id, "text": chunk}
                            outputs[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                            record["samples"] += 1
                        counts[f"{split}_books"] += 1
                        counts[f"{split}_samples"] += record["samples"]
                        counts["accepted_words"] += details["words"]
                        if args.write_books:
                            target = run / "clean" / f"{book.id}.txt"
                            target.parent.mkdir(exist_ok=True)
                            target.write_text(clean, encoding="utf-8")
                except (CorpusError, ValueError) as exc:
                    record.update(status="failed", error=str(exc))
                counts[record["status"]] += 1
                records.append(record)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                print(f"[{index}/{len(books)}] {book.id}: {record['status']}" + (f" — {record['error']}" if "error" in record else ""), flush=True)
    finally:
        for output in outputs.values():
            output.close()
    report = {"version": VERSION, "requested": len(books), "counts": dict(counts), "complete": not (counts["failed"] or counts["needs_review"]), "sources_sha256": sha256(args.sources.read_bytes()), "rules_sha256": sha256(args.overrides.read_bytes()), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    write_json(run / "report.json", report)
    write_audit(run / "audit.html", records, report)
    write_json(args.output / "latest.json", {"run": str(run.resolve()), "complete": report["complete"]})
    print(f"Outputs: {run}\nAccepted: {counts['accepted']}; review: {counts['needs_review']}; failed: {counts['failed']}; duplicates: {counts['duplicate']}", flush=True)
    return 0 if report["complete"] else 1


def write_audit(path: Path, records: list[dict], report: dict) -> None:
    escape = lambda value: html.escape(str(value))
    with path.open("w", encoding="utf-8") as out:
        out.write('<!doctype html><meta charset="utf-8"><title>Corpus cleaning audit</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:16px}article{border-top:1px solid #bbb;padding:20px 0}summary{cursor:pointer}</style><h1>Corpus cleaning audit</h1>')
        out.write("<p>Only accepted books appear in training JSONL. Automated checks do not certify editorial completeness. Review openings, endings, and flagged material.</p>")
        out.write(f"<pre>{escape(json.dumps(report['counts'], indent=2))}</pre>")
        for record in records:
            out.write(f"<article><h2>{escape(record['id'])}: {escape(record.get('source_note', ''))}</h2><p>Status: {escape(record['status'])}</p>")
            for name in ("error", "problems", "opening", "closing"):
                if record.get(name):
                    out.write(f"<h3>{escape(name.title())}</h3><pre>{escape(record[name])}</pre>")
            out.write(f"<details><summary>Provenance and removals</summary><pre>{escape(json.dumps(record, ensure_ascii=False, indent=2))}</pre></details></article>")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path(__file__).with_name("corpus-list.py"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus"))
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="Gutenberg collection root (uses the official PGLAF mirror by default)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between requests (default: 2)")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+", help="Process selected ebook IDs")
    parser.add_argument("--offline", action="store_true", help="Use verified local downloads only")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--cache-raw", action="store_true", help="Keep original downloads on disk (off by default; implied by --download-only)")
    parser.add_argument("--write-books", action="store_true", help="Also write one text file per accepted/review book (off by default to save space)")
    parser.add_argument("--preserve-linebreaks", action="store_true", help="Do not unwrap prose lines")
    parser.add_argument("--overrides", type=Path, default=Path(__file__).with_name("cleaning_rules.json"))
    parser.add_argument("--keep-book-matter", action="store_true", help="Retain original front/back matter and footnotes; still remove digital boilerplate")
    parser.add_argument("--chunk-words", type=int, default=1024, help="Maximum whitespace-delimited words per training sample; not a tokenizer count")
    parser.add_argument("--seed", default="gutenberg-corpus-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or not math.isfinite(args.timeout) or args.delay < 0 or args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit <= 0) or args.chunk_words <= 0:
        parser.error("Delay/retries must be nonnegative; timeout, limit, and chunk-words must be positive.")
    if not (0 <= args.validation_fraction < 1 and 0 <= args.test_fraction < 1 and args.validation_fraction + args.test_fraction < 1):
        parser.error("Split fractions must be nonnegative and sum to less than 1.")
    if urlparse(args.mirror).scheme != "https":
        parser.error("--mirror must use HTTPS.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        books = load_books(args.sources)
        if args.ids:
            unknown = set(args.ids) - {b.id for b in books}
            if unknown:
                raise CorpusError(f"Unknown book IDs: {sorted(unknown)}")
            books = [b for b in books if b.id in args.ids]
        if args.limit:
            books = books[:args.limit]
        downloader = Downloader(args.output, args.mirror, args.delay, args.timeout, args.retries, args.offline, args.cache_raw or args.download_only)
        if args.download_only:
            failures = []
            for index, book in enumerate(books, 1):
                try:
                    data, _ = downloader.fetch(book)
                    print(f"[{index}/{len(books)}] {book.id}: {len(data):,} bytes", flush=True)
                except (CorpusError, OSError, ValueError) as exc:
                    failures.append({"id": book.id, "error": str(exc)})
                    print(f"[{index}/{len(books)}] {book.id}: ERROR {exc}", file=sys.stderr, flush=True)
            write_json(args.output / "download_report.json", {"requested": len(books), "downloaded": len(books) - len(failures), "failures": failures})
            return 1 if failures else 0
        return prepare(books, downloader, args)
    except (CorpusError, OSError, ValueError, SyntaxError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
