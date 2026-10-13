# FemaleAuthorsScratch

**Google Colab copy/paste file: [`COLAB_ALL_IN_ONE.py`](COLAB_ALL_IN_ONE.py).** It contains everything in 13 numbered cells; no ZIP or separate uploads are needed.

FemaleAuthorsScratch trains a generative text model from scratch on a growing corpus of works by female authors. The default configuration has **111,008,256 trainable parameters (approximately 111M)** when the tokenizer reaches its 32,768-token vocabulary target. Both the GPT-2 language model and its byte-level BPE tokenizer are trained from scratch.

| Configuration | Default |
| --- | --- |
| Model | `GPT2LMHeadModel` with randomly initialized weights |
| Transformer layers | 12 |
| Attention heads | 12 per layer; 64 dimensions per head |
| Hidden / feed-forward width | 768 / 3,072 |
| Context window | 1,024 tokens |
| Vocabulary | Up to 32,768 tokens, including special tokens |
| Dropout | 0.1 for embeddings, attention, and residual connections |
| Effective batch | Up to 64 windows per optimizer update: microbatch 8 × accumulation 8 |
| Training hardware / precision | One A100 GPU; BF16 mixed precision with TF32 enabled |
| Optimizer / schedule | AdamW at 3e-4; 3% warmup, cosine decay, 3 epochs |
| Session budget | Two hours of training per session, with resumable checkpoints |
| Progress samples | Three fixed prompts every 100 optimizer updates; printed and saved |

The output language-model head shares its weights with the token embeddings, so those parameters are counted once. With the default layer widths and context length, the exact count is **85,842,432 + 768 × actual vocabulary size**. A smaller learned vocabulary yields fewer parameters. Adding books changes the training data and step count; model size changes only when the learned vocabulary size or architecture changes. Defaults are defined in [`training_config.json`](training_config.json), and [`train_gpt2.py`](train_gpt2.py) reports the actual parameter count before training.

`prepare_corpus.py` reads the current entries in `corpus-list.py`, downloads their plain-text editions, removes digital boilerplate and identifiable non-narrative material, and writes model-training JSONL. Requires **Python 3.10+**, with no third-party dependencies. The input Python file is parsed, never executed.

**Storage defaults:** each download is held in memory and processed individually. Raw books and separate cleaned-book copies are **not saved** unless requested. The final JSONL corpus still requires disk space. Each invocation creates a new output run; previous runs are retained.

## Start in Google Colab — one copy/paste file

1. Open **[`COLAB_ALL_IN_ONE.py`](COLAB_ALL_IN_ONE.py)**. Create a Google Colab notebook and select **Runtime → Change runtime type → A100 GPU**.
2. Copy each numbered `CELL` section into its own code cell, including the comment heading, and run them in order. The format follows the labeled sections in the Poetry50M and FeminineClassicsFinetunedSLM scripts.
3. Edit the corpus in **Cell 2**, cleaning rules in **Cell 3**, and model/training/sample settings in **Cell 4**. Set your run name and session options in **Cell 1**.
4. **Cell 11** prepares the corpus and shows audit findings; **Cell 12** trains or resumes and prints samples every 100 optimizer updates; **Cell 13** generates text from the saved model.

The existing source files are embedded verbatim. Their cells automatically write the modules and configuration into the Drive project directory, preserving the original imports, command-line behavior, and checkpoint fingerprints. You only copy from this one file. Run the complete source-writing cells, keeping their surrounding raw-string quotes and `write_text` calls. These cells write their displayed source/configuration into Drive, so carry over any prior edits made directly to the Drive copies before rerunning them. Existing source modules and the notebook remain available as alternative entry points.

## Alternate: existing notebook with ZIP upload

1. Run `python3 make_colab_bundle.py` in this directory. It packages the **current** corpus list, cleaning rules, configuration, and scripts into `colab_project.zip`; it does not include books or existing checkpoints.
2. Open `train_gpt2_colab.ipynb` in Google Colab and choose **Runtime → Change runtime type → A100 GPU**.
3. Run the notebook cells. Mount Google Drive and upload `colab_project.zip` when prompted.
4. Inspect the preparation audit. Resolve flagged editions with `cleaning_rules.json`, or explicitly set `ALLOW_PARTIAL_CORPUS=True` to train on accepted books only. An incomplete corpus is rejected by default.
5. Train and sample text. Rerun the training cell after reconnecting to resume the latest complete checkpoint.

Drive stores the project under `MyDrive/GPT2Corpus/project`, the corpus under `MyDrive/GPT2Corpus/corpus`, and experiments under `MyDrive/GPT2Corpus/runs/<RUN_NAME>`. Token arrays are cached on Colab's local disk and can be rebuilt from the same corpus and saved tokenizer. Keep the Drive corpus and run directory to resume. Allow several GB of free space: each default-model checkpoint includes weights and two Adam optimizer moments; the newest two and the best checkpoint are retained.

The notebook and scripts are ready for an A100, but **an A100 training run has not been performed here**. Local verification uses a tiny CPU model and invented books. The two-hour default is a per-session training budget, not a promised completion time. Evaluation, checkpoint writing, and export can run past it. Colab can disconnect earlier, and Drive writes can be slow. See [Colab's runtime and Drive guidance](https://research.google.com/colaboratory/faq.html).

## One corpus definition, any number of books

For the single-file Colab workflow, edit the `sources` list in **Cell 2 of `COLAB_ALL_IN_ONE.py`**. For the modular workflow, continue editing **`corpus-list.py`**. Neither the cleaner nor trainer depends on the list's length, order, or particular titles. `corpus_registry.py` is the shared parser/identity logic, and reads the list as Python literals without executing it. Existing URL strings and inline comments continue to work. Optional dictionary entries give you stable grouping without relying on comments:

```python
sources = [
    "https://www.gutenberg.org/files/123/123-0.txt",  # Title by Author
    {"url": "https://www.gutenberg.org/ebooks/456",
     "work_id": "author-work", "note": "Volume one"},
    {"url": "https://www.gutenberg.org/cache/epub/789/pg789.txt",
     "work_id": "author-work", "note": "Volume two"},
]
```

These are schema examples, not additions to your actual list. Use the same `work_id` for volumes, alternate editions, or related material that must stay in one split. URLs must identify Gutenberg books; IDs are derived from them. Repeated IDs are deduplicated, and conflicting explicit groups are rejected. Existing comments infer volume groups as before; explicit IDs are more reliable. Keep comments/work IDs stable to retain existing split assignments when appending books. Changing a comment-derived identity can change that work's split.

After expanding the list or revising rules in the single-file workflow, rerun Cells 2–3 with your edits, set `REBUILD_CORPUS=True` in Cell 1, and choose a **new `RUN_NAME`**. For the alternate ZIP workflow, rebuild the zip and set `UPLOAD_UPDATED_PROJECT=True` and `REBUILD_CORPUS=True` instead. Cached originals are reused and new books are downloaded. A fresh tokenizer and randomly initialized model are trained for the new experiment. A changed corpus is deliberately incompatible with resuming an old checkpoint.

## Training behavior

`train_gpt2.py` constructs **`GPT2LMHeadModel(GPT2Config(...))`** with random parameters. A fresh byte-level BPE tokenizer is trained **only on the training books**; no pretrained weights or tokenizer are downloaded. The architecture uses GPT-2's causal language-model head. See [Hugging Face's GPT-2 configuration](https://huggingface.co/docs/transformers/model_doc/gpt2) and [BPE trainer reference](https://huggingface.co/docs/tokenizers/api/trainers).

Edit the settings in **Cell 4** of the single-file version (or `training_config.json` in the modular version) before a new run:

| Setting | Default |
| --- | --- |
| Architecture | 12 layers, 12 attention heads, 768 hidden width |
| Tokenizer | Up to 32,768 byte-BPE tokens; actual vocabulary recorded |
| Context | 1,024 tokens |
| Microbatch / accumulation | 8 windows × 8 microbatches per optimizer update |
| Schedule | 3 epochs, or `max_steps` if smaller; actual steps derived from data |
| Optimizer | AdamW, learning rate 3e-4, weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 3% warmup followed by cosine decay |
| Precision | BF16 autocast on CUDA, TF32 matrix multiplication |
| Memory / attention | Gradient checkpointing and PyTorch SDPA |
| Checkpoints / validation | Every 100 / 200 updates; also validate and save at epoch/session end |
| Progress samples | Every 100 optimizer updates; three prompts, up to 128 new tokens each |
| Session budget | 7,200 seconds; `--max-seconds 0` disables it |

This is GPT-2 small's transformer shape, with a corpus-specific vocabulary (roughly 111M parameters at the full default vocabulary, rather than the original vocabulary's parameter count). The program prints the actual parameter and token counts, planned optimizer steps, training loss, validation loss/perplexity, and observed tokens per second. BF16 and TF32 require suitable hardware; this script checks CUDA/BF16 support. It also supports `--device cpu` for tiny development tests. The mixed-precision behavior follows [PyTorch's autocast API](https://docs.pytorch.org/docs/stable/amp.html).

All accepted chunks are reassembled into whole books and checked against the manifest's text hashes **before training**. The tokenizer sees only training books. The same frozen tokenizer encodes validation and test. No chunks are independently re-split. Work IDs and exact text hashes are checked for leakage across splits. Near-duplicate editions still require explicit grouping and review.

Every book starts with a BOS token and ends with EOS. Token windows stay within a book, overlap by exactly one context token, and include their short final window. Every text/EOS target appears once per epoch: no lost remainder at mapping or batch boundaries. The pad token has a distinct ID, and its labels are `-100`, so genuine EOS labels remain trainable. Gradient accumulation and evaluation weight losses by actual target-token counts, including short batches. Perplexity is token-level and is not directly comparable across different tokenizers.

Training needs nonempty train and validation sets; tiny trial lists may not produce both under the stable hash split. Add works or explicitly choose a different preparation seed/fraction for that experiment. Do not split samples from the same book into training and validation. Test can be empty; that is reported as `null` instead of inventing a score. Otherwise it is evaluated on the best validation checkpoint only after the full schedule ends.

Three epochs is an initial experiment, not a quality guarantee. A small literary corpus may overfit this architecture well before it learns fluent prose. Watch validation loss; consider a smaller configuration (for example 6 layers, width 384, 6 heads) as a separate experiment. Increasing epochs cannot substitute for more varied data. Changing configuration or the total schedule requires a new run directory.

## Watch training progress

At optimizer steps **100, 200, 300, …**, `generate_samples()` in `train_gpt2.py` prints continuations from the current training weights directly in the Colab cell output. One optimizer step includes all configured gradient-accumulation microbatches. The default prompts are:

- `The evening was quiet, and`
- `She opened the letter and`
- `"I never expected," she said,`

The same prompts and random seeds are reused so you can compare checkpoints. Each continuation uses up to 128 new tokens, temperature 0.8, and top-p 0.95. Generation temporarily uses evaluation mode and restores the training mode and CPU/CUDA random states afterward. Sampling does not change dropout randomness or optimizer updates, but takes additional session time. Very early outputs may be incoherent.

Samples are also appended to **`samples.jsonl`** in the run directory on Drive, with the optimizer step, prompt, continuation, generation settings, and generated token count. Prompts that exceed the available context are shortened from the left and marked in the output. Edit the `sample_*` fields in Cell 4 (or `training_config.json`) to change prompts, cadence, length, seed, temperature, or top-p; `sample_every: 0` disables previews. These observation settings can change on resume without restarting training, and the current settings are saved in `sampling_config.json`.

## Command-line training

Cleaning uses only Python's standard library. Training requires the packages in `requirements-colab.txt`. Install these in an environment with CUDA-enabled PyTorch for an A100. Transformers/Tokenizers are pinned to the API pair used in local tests; Colab's existing compatible PyTorch is retained.

```sh
python3 -m pip install -r requirements-colab.txt

# Unified preparation + tokenization + random initialization + training:
python3 train_gpt2.py --prepare --corpus data/corpus --output runs/experiment-001

# Or train an existing audited run:
python3 train_gpt2.py --corpus data/corpus --output runs/experiment-001

# Later sessions must use the same corpus, config, code, and package versions:
python3 train_gpt2.py --corpus data/corpus --output runs/experiment-001 --resume

python3 generate.py --model runs/experiment-001/best_model \
  --prompt "The evening was quiet, and" --max-new-tokens 200
```

Use a new output directory for the first command if you already ran the second (and vice versa). `--prepare` intentionally cannot be combined with `--resume`. To review preparation settings, run `python3 prepare_corpus.py --help`; all of its existing options remain available separately. `--max-steps` is a **total** schedule cap, not additional steps for each resume. The session wall-clock budget and `sample_*` preview settings can change between resumes without changing the objective/schedule.

Output files:

| Path | Purpose |
| --- | --- |
| `run.json` | Configuration, versions, data/code/tokenizer fingerprints |
| `tokenizer/` | Frozen tokenizer used for every split and every resume |
| `provenance/` | Cleaning report/manifest, source/rules snapshots, implementation |
| `training_plan.json` | Actual token, book, parameter, window, and step counts |
| `metrics.jsonl` | Training and validation measurements |
| `samples.jsonl` | Generated continuations at each sampling step |
| `sampling_config.json` | Preview settings used in the latest session |
| `checkpoint-<step>/` | Model, optimizer, scheduler, RNG and epoch cursor; `complete.json` marks a finished write |
| `best_model/` | Self-contained inference model and tokenizer, exported on graceful completion/session stop |
| `result.json` | Completion flag, best validation step, budget status, final test metrics |

Checkpoints are written into a temporary directory and marked complete before becoming resumable. Incomplete writes are ignored. Sudden runtime loss or manual interruption can lose updates since the last completed checkpoint; rerunning resumes from that checkpoint. The best model uses validation loss, while resume always uses the latest completed optimizer state. Resume restores Python/NumPy/PyTorch RNG state and the shuffle cursor. GPU kernels can be nondeterministic; bitwise A100 equivalence is not promised. Only resume your own checkpoints: optimizer/RNG files use PyTorch pickle serialization. `best_model` uses safetensors.

Checkpoints from the immediately preceding trainer (before progress sampling was added) are recognized and can resume with this version. All training settings, corpus/tokenizer checksums, data code, and package versions must still match; arbitrary code changes remain incompatible.

If the installed versions change in a future Colab image, the script refuses a mismatched resume. `run.json` records the original versions so you can restore a compatible environment. A CUDA out-of-memory error requires lowering `batch_size` in a new experiment; increase `gradient_accumulation` if you want to maintain the nominal effective batch size. No external experiment tracker or model publishing is enabled.

## Start small

Run the synthetic tests without downloading anything:

```sh
python3 -m unittest discover -s tests -v
```

Optionally process just one actual book:

```sh
python3 prepare_corpus.py --limit 1 --output data/trial
```

Or select particular IDs from the list:

```sh
python3 prepare_corpus.py --ids 1342 --output data/trial
```

When ready to process the entire list, run `python3 prepare_corpus.py`. Use `--output /path/to/another/drive/corpus` to choose a different disk. No full-corpus run was performed during development.

## Cleaning behavior

The default is **main narrative**, keeping chapter headings and epigraphs:

- Require unambiguous Gutenberg start/end markers; strip both external wrappers and their licenses/credits.
- Remove bracketed illustration captions, clearly delimited transcriber notes (singular/plural), labelled footnotes, explicit page labels, and paired underscore emphasis markers. Preserve literal underscores inside words.
- Locate a plausible first chapter using the surrounding prose, skipping tables of contents. Preserve a labelled prologue/prelude before that chapter. Ambiguous starts, including unchaptered stories, require a rule.
- Normalize Unicode to NFC and line endings to LF; retain accents, case, original spelling, and punctuation. No lossy decoding, lowercasing, stopword removal, or blanket punctuation stripping.
- Unwrap clearly identifiable prose paragraphs while preserving short/indented lines that may be verse. Keep printed hyphens rather than guessing whether they are discretionary. `--preserve-linebreaks` disables unwrapping.
- Detect remaining Gutenberg boilerplate, transcriber text, contents headings, publisher material, footnote references, markup, suspect encoding, excessive removal, and unexpectedly short texts.

Use `--keep-book-matter` to retain original prefaces, dedications, contents, and footnotes. Digital wrappers, illustration placeholders, and identified transcriber notes are still removed. Explicit `remove` rules still apply in this mode, but narrative start/end rules do not.

**“Completely clean” cannot be certified by generic text rules.** Editions differ, and a chapter heuristic can mistake editorial prose for a book's beginning; unmarked introductions, advertisements, OCR mistakes, and unusual footnotes may evade detection. An `accepted` status means automated checks passed, not that someone reviewed the complete book. Review the audit and use exact per-edition rules when strict narrative purity is required. Flagged books are excluded from training rather than silently included.

## Outputs

Find the current run directory in `<output>/latest.json`. Each run contains:

| File | Content |
| --- | --- |
| `train.jsonl` | Accepted training samples |
| `validation.jsonl` | Accepted validation samples |
| `test.jsonl` | Accepted test samples |
| `manifest.jsonl` | One record per requested book: status, source URLs, source checksum, metadata, removals, findings, and text previews |
| `report.json` | Counts, settings, input/rules checksums, and completion status |
| `audit.html` | Local, readable audit of openings, endings, removals, and findings |
| `corpus-list.py`, `cleaning_rules.json` | Snapshots of the editable inputs used for this run |

Use `--write-books` to additionally save `clean/<id>.txt` and flagged candidates in `review/<id>.txt`. This duplicates text and is off by default. Failed downloads have no candidate file.

Each JSONL line is an independent record:

```json
{"id":"1342:0","book_id":"1342","work_id":"pride and prejudice by jane austen","text":"CHAPTER I.\n\n..."}
```

Samples contain at most 1,024 whitespace-delimited words by default; change this with `--chunk-words`. This is **not a tokenizer count**. Splits occur at word boundaries and may cut sentences or paragraphs. Samples never cross books or overlap, and concatenating a book's samples restores its cleaned text exactly, including the final short sample. `training_data.py` reconstructs and verifies each book before tokenization and handles context limits and BOS/EOS boundaries.

Splits are deterministically assigned by work identity before chunking, approximately 90% train / 5% validation / 5% test. Volumes identified from the source comments stay together. Exact cleaned-text duplicates are excluded; this does not detect abridgments, adaptations, or other near-duplicates. `work_id` overrides can group related works. Tiny trials may have empty validation/test files. Set both `--validation-fraction 0 --test-fraction 0` for an entirely training-only export.

Exit code `0` means every requested book was accepted or identified as an exact duplicate. Exit code `1` means a failure or unresolved review item; successful books are still exported. An interrupted run does not replace `latest.json`, and may leave a partial run directory.

## Reviewing a difficult edition

Keep only the single edition needed for review:

```sh
python3 prepare_corpus.py --ids 1342 --output data/review-one --cache-raw --write-books
```

Read its audit and original `data/review-one/raw/1342.txt`. The manifest and `raw/1342.json` contain its `sha256`. Add an entry to `cleaning_rules.json`, using that exact SHA-256. The following is a **schema example, not a tested rule for a real book**:

```json
{
  "version": 1,
  "books": {
    "123": {
      "raw_sha256": "REPLACE_WITH_THE_64_CHARACTER_SOURCE_SHA256",
      "start": {"text": "CHAPTER I.", "occurrence": 2},
      "end": {"text": "PUBLISHER'S CATALOGUE"},
      "remove": [{"text": "[1]", "count": 3}],
      "work_id": "an-invented-work",
      "notes": "The first chapter heading appears in the contents."
    }
  }
}
```

- Anchors are **literal strings**, not regexes. `start` is inclusive and `end` exclusive. Both are optional; omit `end` when the Gutenberg end marker already bounds the narrative.
- Anchors match after NFC/newline normalization, wrapper removal, and explicit illustration/transcriber artifact removal, but before prose reflow and underscore removal.
- An anchor must occur exactly once unless a 1-based `occurrence` is supplied. Every literal removal must match its declared count (default 1), or the book fails rather than guessing.
- Rules are tied to the raw-byte SHA-256 and fail if Gutenberg changes the edition. Unspecified books use automatic cleaning.
- `accept_findings` can list confirmed false-positive finding names shown under `problems`, such as `possible_footnote` when a bracketed number is genuinely narrative. Gutenberg residue, invalid control characters, replacement characters, and HTML markup cannot be waived. Prefer an actual boundary/removal correction over waiving a finding.

Then reprocess the cached single book without network access:

```sh
python3 prepare_corpus.py --ids 1342 --output data/review-one --offline --write-books
```

Caching is optional. `--download-only` implies `--cache-raw` and explicitly saves originals without cleaning. Without caching, later runs download again. Cached sources are reused only when their ID, source URL, and checksum match. To refresh a changed edition, use a fresh output directory.

## Downloads and provenance

The default transport maps each supplied Gutenberg ID to the same ID on the [officially listed PGLAF mirror](https://www.gutenberg.org/MIRRORS.ALL), using the [Gutenberg mirror directory layout](https://www.gutenberg.org/help/mirroring.html). This avoids crawling catalog pages. Original URLs remain in the manifest. Configure another HTTPS collection root with `--mirror` if necessary.

Requests are sequential, spaced two seconds apart, with timeouts and bounded retries. Missing UTF-8 files fall back to generated text or older plain-text editions on the same mirror. Legacy encodings are decoded without replacement and flagged for review. HTTP access refusals stop further network requests. Downloads are limited to 30 MiB per book; HTML error responses, incomplete marker pairs, and conflicting numeric ebook IDs are rejected.

Raw originals are retained only with `--cache-raw`. The manifest records removal lengths, hashes, and short previews rather than storing a second full copy of discarded material. Source notes are provenance and are never inserted into training text. Keep the manifest for traceability; inspect original source terms if you plan to redistribute a derived corpus.

## Validation performed

Tests use tiny invented texts and mocked HTTP responses. With training dependencies installed, they also train a tiny randomly initialized GPT-2, resume it, compare its final weights exactly against an uninterrupted CPU run, and generate text from its export. They also verify the 100-update sample cadence, context limits, fixed-seed previews, RNG/mode restoration, and identical CPU training weights with and without previews. They check train-only tokenizer fitting, Unicode round trips, complete token-target coverage, EOS/padding labels, split leakage rejection, tampered chunks, portable run pointers, and resume configuration checks. They cover source-list parsing without execution, wrapper validation, TOC/narrative boundaries, prologues, footnotes, Unicode and verse preservation, literal rule checks, cache integrity, HTTP fallback/retry behavior, exact deduplication, volume grouping, lossless chunking, and an end-to-end offline run including review and download failures.

The full corpus has **not** been downloaded or manually audited. No claim of edition-by-edition completeness is made.

## Earlier repository notes

<details>
<summary>Original project description and training notes</summary>

# FemaleAuthorsScratch
A model trained from scratch on a corpus of female works from the 18th and 19th centuries using a randomly initialized GPT 2 config. 

Once finished training, it's looking to be around 111M in size. I might put it on Hugging Face. In the future, I might experiment with further splitting (and perhaps enhancing) this corpus between anonymous female authors/female authors with male pseudonyms and female authors who wrote and published as women while they were alive. 

## Results: 

## Training Logs/Other Information: 
After training on a smaller dataset (around 72 books) for a few epochs, I decided to put the actual run in abeyance while gathering more books. I also have to wait for my compute units to refresh, so you will not be seeing anything from this repository for quite some time. 

## Comparison: Female Fine-tune vs. Female Authors From Scratch

</details>
