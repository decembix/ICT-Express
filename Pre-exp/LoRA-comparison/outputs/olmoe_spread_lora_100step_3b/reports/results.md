# SPREAD router/LoRA comparison

All adaptation rows are teacher-free. WikiText reports perplexity (lower is better); PIQA reports normalized accuracy (higher is better). Frozen baselines are listed under `none` because they do not consume a training dataset.

## 3B

| Train data | Method | WikiText PPL ↓ | PIQA AccNorm ↑ | Trainable | Total | C→K |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| none | Router frozen / Expert frozen | 1583.96 | 52.12% | 0 | 2,992,015,360 | 32→25 |
| none | No router / SPREAD uniform concat | 11373.10 | 51.63% | 0 | 2,991,196,160 | 32→25 |
| piqa | Router+LoRA / Expert frozen | 1583.49 | 52.18% | 265,344 | 2,992,280,704 | 32→25 |
| piqa | Router+LoRA / Expert+LoRA | 1585.75 | 52.88% | 29,756,544 | 3,021,771,904 | 32→25 |
| piqa | Router frozen / Expert+LoRA | 1584.24 | 52.34% | 29,491,200 | 3,021,506,560 | 32→25 |
| piqa | No router / concat FFN+LoRA | 9630.14 | 51.74% | 10,616,832 | 3,001,812,992 | 32→25 |
| piqa | Low-rank router / Expert frozen | 11371.53 | 51.63% | 265,344 | 2,991,461,504 | 32→25 |
| piqa | Low-rank router / Expert+LoRA | 11903.17 | 51.03% | 29,756,544 | 3,020,952,704 | 32→25 |
| wikitext | Router+LoRA / Expert frozen | 1480.16 | 52.61% | 265,344 | 2,992,280,704 | 32→25 |
| wikitext | Router+LoRA / Expert+LoRA | 171.34 | 54.08% | 29,756,544 | 3,021,771,904 | 32→25 |
| wikitext | Router frozen / Expert+LoRA | 178.90 | 53.43% | 29,491,200 | 3,021,506,560 | 32→25 |
| wikitext | No router / concat FFN+LoRA | 337.18 | 51.52% | 10,616,832 | 3,001,812,992 | 32→25 |
| wikitext | Low-rank router / Expert frozen | 9927.43 | 51.74% | 265,344 | 2,991,461,504 | 32→25 |
| wikitext | Low-rank router / Expert+LoRA | 1187.07 | 51.80% | 29,756,544 | 3,020,952,704 | 32→25 |
