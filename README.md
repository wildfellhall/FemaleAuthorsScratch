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
| Optimizer / schedule | AdamW, betas (0.9, 0.95), weight decay 0.1; 3% warmup, cosine decay from 3e-4 toward 3e-5 |
| Training length | Up to 12 epochs; stop after 3 full epochs without a validation-loss improvement of 0.002 nats/token |
| Session budget | No fixed time limit; optional `MAX_SESSION_SECONDS`, with resumable local checkpoints |
| Gradient checkpointing | Off by default; enable if GPU memory is tight |
| Validation | Stride-512 context windows; loss, perplexity, and bits per byte; best checkpoint selected |
| Progress samples | Three fixed prompts every 100 optimizer updates; printed and saved |

A model trained from scratch on a corpus of female works from the 18th and 19th centuries using a randomly initialized GPT 2 config. 

Once finished training, it's looking to be around 111M in size. I might put it on Hugging Face. In the future, I might experiment with further splitting (and perhaps enhancing) this corpus between anonymous female authors/female authors with male pseudonyms and female authors who wrote and published as women while they were alive. 

## Results: 

## Training Logs/Other Information: 
After training on a smaller dataset (around 72 books) for a few epochs, I decided to put the actual run in abeyance while gathering more books. I also have to wait for my compute units to refresh, so you will not be seeing anything from this repository for quite some time. 

## Comparison: Female Fine-tune vs. Female Authors From Scratch

## Notes/Experiment and Explanation: 
