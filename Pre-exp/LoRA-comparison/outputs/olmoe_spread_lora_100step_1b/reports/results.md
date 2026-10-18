# SPREAD router/LoRA comparison

All adaptation rows are teacher-free. WikiText reports perplexity (lower is better); PIQA reports normalized accuracy (higher is better). Frozen baselines are listed under `none` because they do not consume a training dataset.

## 1B

| Train data | Method | WikiText PPL ↓ | PIQA AccNorm ↑ | Trainable | Total | C→K |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| none | Router frozen / Expert frozen | 12541.01 | 50.00% | 0 | 1,280,182,272 | 10→8 |
| none | No router / SPREAD uniform concat | 12516.41 | 49.56% | 0 | 1,279,920,128 | 10→8 |
| piqa | Router+LoRA / Expert frozen | 12560.41 | 50.38% | 263,168 | 1,280,445,440 | 10→8 |
| piqa | Router+LoRA / Expert+LoRA | 12557.65 | 49.84% | 9,700,352 | 1,289,882,624 | 10→8 |
| piqa | Router frozen / Expert+LoRA | 12538.56 | 49.29% | 9,437,184 | 1,289,619,456 | 10→8 |
| piqa | No router / concat FFN+LoRA | 12314.51 | 50.05% | 3,932,160 | 1,283,852,288 | 10→8 |
| piqa | Low-rank router / Expert frozen | 12525.41 | 49.35% | 263,168 | 1,280,183,296 | 10→8 |
| piqa | Low-rank router / Expert+LoRA | 12477.90 | 49.13% | 9,700,352 | 1,289,620,480 | 10→8 |
| wikitext | Router+LoRA / Expert frozen | 11948.05 | 49.84% | 263,168 | 1,280,445,440 | 10→8 |
| wikitext | Router+LoRA / Expert+LoRA | 2302.48 | 50.82% | 9,700,352 | 1,289,882,624 | 10→8 |
| wikitext | Router frozen / Expert+LoRA | 2320.82 | 50.11% | 9,437,184 | 1,289,619,456 | 10→8 |
| wikitext | No router / concat FFN+LoRA | 1735.09 | 49.46% | 3,932,160 | 1,283,852,288 | 10→8 |
| wikitext | Low-rank router / Expert frozen | 11361.13 | 49.40% | 263,168 | 1,280,183,296 | 10→8 |
| wikitext | Low-rank router / Expert+LoRA | 2328.93 | 48.97% | 9,700,352 | 1,289,620,480 | 10→8 |
