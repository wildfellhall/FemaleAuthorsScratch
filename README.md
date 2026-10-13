# FemaleAuthorsScratch

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

A model trained from scratch on a corpus of female works from the 18th and 19th centuries using a randomly initialized GPT 2 config. 

Once finished training, it's looking to be around 111M in size. I might put it on Hugging Face. In the future, I might experiment with further splitting (and perhaps enhancing) this corpus between anonymous female authors/female authors with male pseudonyms and female authors who wrote and published as women while they were alive. 

## Results: 

## Training Logs/Other Information: 
After training on a smaller dataset (around 72 books) for a few epochs, I decided to put the actual run in abeyance while gathering more books. I also have to wait for my compute units to refresh, so you will not be seeing anything from this repository for quite some time. 

## Comparison: Female Fine-tune vs. Female Authors From Scratch

</details>
