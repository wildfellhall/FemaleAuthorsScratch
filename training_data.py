"""Audited corpus -> train-only byte BPE -> lossless, book-bounded token windows."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from corpus_registry import CorpusError
from prepare_corpus import sha256, write_json

SPLITS = ("train", "validation", "test")
DATA_VERSION = 1
SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_run(path: Path) -> Path:
    if (path / "report.json").is_file():
        return path
    latest = path / "latest.json"
    if not latest.is_file():
        raise CorpusError(f"No prepared corpus at {path}; run prepare_corpus.py first.")
    pointer = json.loads(latest.read_text())
    # Relative paths survive copying a preparation run from a laptop to Colab.
    if pointer.get("relative_run"):
        target = path / pointer["relative_run"]
    else:
        target = path / "runs" / Path(pointer["run"]).name
        if not target.is_dir():
            target = Path(pointer["run"])
    if not (target / "report.json").is_file():
        raise CorpusError(f"Preparation run is missing: {target}")
    return target


def manifest_for(run: Path) -> dict:
    accepted = {}
    works, hashes = {}, set()
    with (run / "manifest.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["status"] != "accepted":
                continue
            book_id, work_id, split = row["id"], row["work_id"], row["split"]
            if split not in SPLITS or book_id in accepted:
                raise CorpusError("Invalid split or repeated accepted book in manifest.")
            if work_id in works and works[work_id] != split:
                raise CorpusError(f"Work {work_id!r} leaks across splits.")
            if row["clean_sha256"] in hashes:
                raise CorpusError("Duplicate accepted text in manifest.")
            works[work_id] = split
            hashes.add(row["clean_sha256"])
            accepted[book_id] = row
    return accepted


def iter_documents(run: Path, split: str, manifest: dict | None = None):
    """Reconstruct and verify one book at a time, including all final chunks."""
    manifest = manifest if manifest is not None else manifest_for(run)
    current, pieces, seen = None, [], set()

    def finish():
        if current is None:
            return None
        record = manifest[current]
        text = "".join(pieces)
        if len(pieces) != record["samples"] or sha256(text.encode("utf-8")) != record["clean_sha256"]:
            raise CorpusError(f"Missing/reordered/changed chunks for book {current}.")
        if not text.strip():
            raise CorpusError(f"Empty accepted book {current}.")
        return {"book_id": current, "work_id": record["work_id"], "text": text}

    with (run / f"{split}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            book_id = row["book_id"]
            if book_id not in manifest or manifest[book_id]["split"] != split or row["work_id"] != manifest[book_id]["work_id"]:
                raise CorpusError(f"Unaccepted book or wrong split/work in {split}: {book_id}")
            if book_id != current:
                document = finish()
                if document:
                    yield document
                if book_id in seen:
                    raise CorpusError(f"Noncontiguous chunks for {book_id}.")
                seen.add(book_id)
                current, pieces = book_id, []
            if row["id"] != f"{book_id}:{len(pieces)}" or row.get("chunk_index", len(pieces)) != len(pieces):
                raise CorpusError(f"Out-of-order chunk for {book_id}.")
            if not isinstance(row["text"], str):
                raise CorpusError("Training text must be a string.")
            pieces.append(row["text"])
    document = finish()
    if document:
        yield document
    expected = {book_id for book_id, row in manifest.items() if row["split"] == split}
    if seen != expected:
        raise CorpusError(f"Missing accepted books in {split}: {sorted(expected - seen)}")


def inspect_corpus(run: Path, allow_partial: bool = False) -> dict:
    report = json.loads((run / "report.json").read_text())
    if not report["complete"] and not allow_partial:
        raise CorpusError("Corpus has review items/download failures. Inspect audit.html; use --allow-partial-corpus to explicitly train only accepted books.")
    manifest = manifest_for(run)
    counts = {}
    for split in SPLITS:
        count, chars = 0, 0
        for document in iter_documents(run, split, manifest):
            count += 1
            chars += len(document["text"])
        counts[split] = {"books": count, "characters": chars}
    if not counts["train"]["books"] or not counts["validation"]["books"]:
        raise CorpusError("Training needs nonempty train and validation splits. Add works or choose a different preparation --seed / split fraction; never split chunks of the same work across sets.")
    hashes = {name: file_sha256(run / name) for name in ("manifest.jsonl", *(f"{s}.jsonl" for s in SPLITS))}
    fingerprint = sha256(json.dumps(hashes, sort_keys=True).encode())
    return {"fingerprint": fingerprint, "files": hashes, "counts": counts, "complete": report["complete"]}


def train_tokenizer(run: Path, destination: Path, vocab_size: int, min_frequency: int):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    backend = ByteLevelBPETokenizer(add_prefix_space=False)
    # Never expose validation/test texts to tokenizer training.
    backend.train_from_iterator(
        (doc["text"] for doc in iter_documents(run, "train")),
        vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS), show_progress=True,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend._tokenizer, bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[0], unk_token=None, pad_token=SPECIAL_TOKENS[1],
        # Full books are encoded before windowing, so don't truncate to model size.
        model_max_length=10**12, padding_side="right",
    )
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(destination)
    return tokenizer


def tokenize_corpus(run: Path, destination: Path, tokenizer) -> dict:
    import numpy as np

    destination.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in SPLITS:
        books, offset = [], 0
        token_path = destination / f"{split}.bin"
        with token_path.open("wb") as out:
            for doc in iter_documents(run, split):
                # Literal special-token strings in a source must not turn into controls.
                if any(token in doc["text"] for token in SPECIAL_TOKENS):
                    raise CorpusError(f"Reserved tokenizer control token in book {doc['book_id']}.")
                ids = tokenizer.encode(doc["text"], add_special_tokens=False, truncation=False)
                ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
                np.asarray(ids, dtype="<u4").tofile(out)
                books.append({"book_id": doc["book_id"], "offset": offset, "length": len(ids)})
                offset += len(ids)
        stats[split] = {"tokens": offset, "target_tokens": offset - len(books), "books": books, "sha256": file_sha256(token_path)}
    write_json(destination / "index.json", stats)
    return stats


class TokenWindows:
    """Memory-mapped windows that never cross books and lose no target tokens.

    A single context token overlaps neighboring windows. BOS predicts the first
    text token; the final text token predicts EOS. Padding is not a target.
    """
    def __init__(self, directory: Path, split: str, context_length: int):
        import numpy as np
        if context_length < 2:
            raise ValueError("context_length must be at least two")
        self.context_length = context_length
        info = json.loads((directory / "index.json").read_text())[split]
        self.books = info["books"]
        self.tokens = np.memmap(directory / f"{split}.bin", mode="r", dtype="<u4") if info["tokens"] else []
        self.ends, count = [], 0
        for book in self.books:
            count += math.ceil((book["length"] - 1) / (context_length - 1))
            self.ends.append(count)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        book_index = bisect_right(self.ends, index)
        book = self.books[book_index]
        local = index - (self.ends[book_index - 1] if book_index else 0)
        start = book["offset"] + local * (self.context_length - 1)
        end = min(start + self.context_length, book["offset"] + book["length"])
        return self.tokens[start:end].astype("int64").tolist()


def collate_windows(windows, pad_token_id: int):
    import torch
    width = max(map(len, windows))
    ids = torch.full((len(windows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, tokens in enumerate(windows):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    # EOS is a real target: pad has its own ID and is masked by attention mask.
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
