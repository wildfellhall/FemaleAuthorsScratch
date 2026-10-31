# Corpus and training update — 2026-10-30

The notebook embeds the 600 entries from [CorpusCollection at f6fe26fd](https://github.com/wildfellhall/CorpusCollection/blob/f6fe26fd2ebdb1d76bc7d591a8f3b3cb8ef8c00c/femalecorpus.py). One URL is corrected: *Sir Tom* is [Gutenberg 30692](https://www.gutenberg.org/ebooks/30692), whereas the source repeats 66376 (*Young Musgrave*). This gives 600 unique edition IDs, up from 499. The list remains directly editable; processing and training schedules use the actual accepted data size.

The live check covered the 100 additions and corrected *Sir Tom*, not all 600 editions. After parser fixes, 90 of these 101 editions pass individual cleaning checks, totaling 9,180,048 words before corpus-level deduplication. The parser now handles wrapped Gutenberg markers, decorated terminal transcriber notes, and empty footnote headings. It also preserves the final chapter of *Sir Tom*, whose title is “The End.”

Eleven additions still need review: 58582, 8954, 34542, 4047, 21617, 31681, 33695, 56787, 22806, 1905, and 10777. Findings include ambiguous narrative boundaries, footnotes, contents/publisher material, and one encoding problem. `audit.json` records findings during notebook execution. Strict preparation stops for review by default; `ALLOW_PARTIAL_CORPUS=True` explicitly permits the accepted subset. Earlier editions may also have findings.

Before cleaning, stable Gutenberg-ID splits contain 549 training, 23 validation, and 28 test editions. Accepted counts and token totals are determined at runtime. Splits remain edition-based; related volumes can land in different sets, so evaluation does not measure generalization to entirely unseen authors or series.

The model stays at 12 layers × 768 width with 12 heads and a training-only 32K BPE vocabulary. Increasing context to 2,048 tokens gives longer prose continuations more preceding text, with 111,794,688 parameters at full vocabulary. Microbatch 4 × accumulation 8 maintains roughly 65K target tokens per update. Longer attention still increases compute; A100 throughput has not been measured.

Keep AdamW at 3e-4 with warmup/cosine decay, dropout 0.1, and validation-based checkpoint selection. Twelve epochs remain an experimental ceiling, with early stopping: [repeated data has diminishing returns](https://arxiv.org/abs/2305.16264), so more epochs do not guarantee improvement. Validation runs every 500 updates and each epoch; samples remain every 100 updates. Changed context and evaluation stride mean losses should not be compared directly with the old recipe.

Start a fresh run: the corpus/parser fingerprint and positional embeddings changed. All 80 tests pass, including training beyond position 1,024 and GGUF/Hugging Face logits agreement on 1,536 tokens. No full A100 training or model-quality benchmark was performed.
