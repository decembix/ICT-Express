# SPREAD router/LoRA comparison

All adaptation rows are teacher-free. WikiText reports perplexity (lower is better); PIQA reports normalized accuracy (higher is better). Frozen baselines are listed under `none` because they do not consume a training dataset.

## 5B

| Train data | Method | WikiText PPL ↓ | PIQA AccNorm ↑ | Trainable | Total | C→K |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| none | Router frozen / Expert frozen | 654.85 | 60.01% | 0 | 5,005,936,640 | 57→45 |
| none | No router / SPREAD uniform concat | 7963.36 | 51.41% | 0 | 5,004,462,080 | 57→45 |
| piqa | Router+LoRA / Expert frozen | 654.79 | 60.45% | 267,904 | 5,006,204,544 | 57→45 |
| piqa | Router+LoRA / Expert+LoRA | 651.76 | 60.50% | 53,352,064 | 5,059,288,704 | 57→45 |
| piqa | Router frozen / Expert+LoRA | 652.29 | 60.34% | 53,084,160 | 5,059,020,800 | 57→45 |
| piqa | No router / concat FFN+LoRA | 10373.88 | 52.77% | 18,481,152 | 5,022,943,232 | 57→45 |
| piqa | Low-rank router / Expert frozen | 7974.92 | 51.47% | 267,904 | 5,004,729,984 | 57→45 |
| piqa | Low-rank router / Expert+LoRA | 7842.12 | 51.85% | 53,352,064 | 5,057,814,144 | 57→45 |
| wikitext | Router+LoRA / Expert frozen | 613.52 | 60.34% | 267,904 | 5,006,204,544 | 57→45 |
| wikitext | Router+LoRA / Expert+LoRA | 56.06 | 65.02% | 53,352,064 | 5,059,288,704 | 57→45 |
| wikitext | Router frozen / Expert+LoRA | 56.15 | 65.07% | 53,084,160 | 5,059,020,800 | 57→45 |
| wikitext | No router / concat FFN+LoRA | 176.01 | 54.90% | 18,481,152 | 5,022,943,232 | 57→45 |
| wikitext | Low-rank router / Expert frozen | 7076.64 | 51.63% | 267,904 | 5,004,729,984 | 57→45 |
| wikitext | Low-rank router / Expert+LoRA | 860.14 | 52.39% | 53,352,064 | 5,057,814,144 | 57→45 |
