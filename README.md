# FemaleAuthorsScratch

| Configuration | Default |
| --- | --- |
| Model | `GPT2LMHeadModel` with randomly initialized weights |
| Parameters | 111,008,256 at the full 32,768-token vocabulary; fewer if the learned vocabulary is smaller |
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

A model trained from scratch on a corpus of female works from the 18th and 19th centuries using a randomly initialized GPT 2 config. 

Once finished training, it's looking to be around 111M in size. I might put it on Hugging Face. In the future, I might experiment with further splitting (and perhaps enhancing) this corpus between anonymous female authors/female authors with male pseudonyms and female authors who wrote and published as women while they were alive. 

## Google Colab

Copy the seven numbered cells in [`main.py`](main.py) into an A100 notebook and run them in order. It follows the direct-cell format of [Poetry50M](https://github.com/wildfellhall/Poetry50M/blob/main/main.py), using the existing custom PyTorch training loop.

- **Cell 1:** settings. **Cell 2:** the editable Python `sources` list and `CLEANING_RULES`. Append books freely; related volumes/editions can share a dictionary entry's `work_id`.
- **Cells 3–4:** download books into memory, clean them, fit the tokenizer on training books, and create token windows. Review flagged editions in `audit.json`; fix their rules in Cell 2 before training.
- **Cells 5–7:** train, print samples every 100 optimizer updates, and generate from the saved model.

No Drive mount, uploaded corpus list, generated helper scripts, or directory setup is required. The script creates one local `female-authors/` output folder for audit results, checkpoints, and `best_model/`; books and token arrays stay in memory. Colab's local files disappear when the runtime is deleted. Cell 7 includes an optional model download; keep the entire output folder if you want to resume later.

Set `RESUME=True` to continue from a local checkpoint with the same corpus and settings. After changing the corpus or model, choose a new `OUTPUT_DIR` and start fresh. Start a new run when switching from the previous file-based workflow; its existing model exports remain usable for inference.

Local checks: `python3 -m unittest discover -s tests -v`. Syntax, cleaning, tiny CPU training, checkpoint resume, and progress sampling are tested. An A100 training run has not been performed here.

## Results: 

## Training Logs/Other Information: 
After training on a smaller dataset (around 72 books) for a few epochs, I decided to put the actual run in abeyance while gathering more books. I also have to wait for my compute units to refresh, so you will not be seeing anything from this repository for quite some time. 

## Comparison: Female Fine-tune vs. Female Authors From Scratch
