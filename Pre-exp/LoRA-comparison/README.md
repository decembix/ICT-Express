# SPREAD OLMoE Router/LoRA Comparison

이 패키지는 `allenai/OLMoE-1B-7B-0924`를 SPREAD로 압축한 뒤 KD 없이
WikiText-2 또는 PIQA의 task loss만으로 LoRA/router를 학습합니다. 압축
backbone은 한 번만 저장하고 각 조건은 작은 adapter state만 저장합니다.

## 비교 조건

| key | Router | Router LoRA | Expert LoRA | 학습 여부 |
|---|---|---:|---:|---:|
| `router_lora` | teacher router를 group별로 합친 frozen base | ✓ | — | ✓ |
| `router_lora_expert_lora` | frozen base | ✓ | group별 ✓ | ✓ |
| `router_frozen` | frozen base | — | — | 평가 전용 |
| `router_frozen_expert_lora` | frozen base | — | group별 ✓ | ✓ |
| `uniform` | 없음; 정확히 1/K | — | — | 기존 SPREAD, 평가 전용 |
| `uniform_expert_lora` | 없음; 정확히 1/K | — | concat FFN ✓ | ✓ |
| `lowrank_router` | full-rank base 없이 `BAx` | 해당 router 자체 | — | ✓ |
| `lowrank_router_expert_lora` | full-rank base 없이 `BAx` | 해당 router 자체 | group별 ✓ | ✓ |

`router_*`와 `lowrank_router_*`도 모든 K개 group을 계산합니다. sparse Top-K를
사용하지 않으므로 uniform concat과 active FFN capacity가 같고, router weighting의
효과만 비교합니다. Low-rank-only router의 B는 0으로 초기화되어 최초 forward는
uniform SPREAD와 같습니다.

Router base의 각 행은 해당 SPREAD group에 속한 teacher expert router 행을
SPREAD의 `m_reap_weights`로 평균해 만듭니다. Expert LoRA는 router 조건에서는
각 group의 gate/up/down projection에 독립적으로 붙고, router-free 조건에서는
concatenated Dense FFN의 세 projection에 붙습니다. 같은 rank라도 trainable
parameter 수가 다르므로 결과 CSV에 이를 반드시 함께 보고합니다.

## 압축 크기

| label | SPREAD C→K | 예상 실제 파라미터 |
|---|---:|---:|
| `1b` | 10→8 | 약 1.280B |
| `3b` | 32→25 | 약 2.991B |
| `5b` | 57→45 | 약 5.004B |

K는 OLMoE에서 목표 크기에 가장 가까운 정수 group 수입니다. C/K는 기존
Table-1의 10/8에 가깝게 유지했습니다. 정확한 크기는 build 직후
`backbones/<size>/compression_metadata.json`에 실측값으로 기록됩니다.

## SPREAD GitHub 코드 의존성

현재 SPREAD commit `0f7568c14bfb0053e2e536d54b01d5de637b34e2`에는 이
실험에 필요한 다음 파일이 Git으로 추적되어 있습니다.

- `common/HEM/hem_core.py`
- `common/modules/{expert_importance,expert_merger,gram_logdet,model_adapters}.py`
- `common/artifacts/wikitext103/*`
- `experiments/olmoe/table1_noKD/build_table1_models.py`

따라서 이 commit을 다른 서버에 clone하면 SPREAD expert selection과 HEM-SVD
group merge를 재생성할 수 있습니다. 반면 기존 `SPREAD/Edge/`는 현재
untracked이므로 clone에 포함되지 않습니다. 이 패키지는 `Edge/`를 import하지
않으며, 시작 시 필요한 SPREAD 파일이 실제 Git-tracked 상태인지 검사합니다.

권장 배치는 두 저장소를 나란히 clone하고 `--spread-root`로 명시하는 것입니다.

```text
workspace/
├── SPREAD/
└── ICT-Express/
    └── Pre-exp/LoRA-comparison/
```

## 학습/evaluation 정의

- WikiText 학습: pinned WikiText-2 train split, causal next-token CE.
- PIQA 학습: train split, 두 답안의 length-normalized continuation log-likelihood에
  대한 supervised cross entropy.
- WikiText 평가: test split perplexity.
- PIQA 평가: validation split `acc_norm`; prompt는 lm-eval과 같은
  `Question: {{goal}}\nAnswer:`입니다.
- KD, teacher logits, teacher hidden-state loss는 사용하지 않습니다.
- frozen baseline 두 개는 학습 dataset과 무관하므로 한 번만 평가해 `none`으로
  저장합니다.

## 설치

```bash
cd /home/xyzysh/ICT-Express/Pre-exp/LoRA-comparison
PY=/home/xyzysh/miniconda3/envs/spread-table1/bin/python

"$PY" -m pip install -e .
"$PY" -m unittest discover -s tests -v

export HF_HOME=/home/xyzysh/work/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs
```

다른 서버에서 teacher/dataset이 캐시에 없으면 `--no-local-files-only`(기본값)를
사용합니다. 모두 내려받은 뒤 완전 offline 실행에는 `--local-files-only`를
추가합니다.

## 먼저 실행할 smoke test

1B, 2 steps, 소량 evaluation으로 구조와 gradient를 확인합니다.

```bash
"$PY" -u run_experiment.py \
  --phase all \
  --spread-root /home/xyzysh/work/SPREAD \
  --output-root "$PWD/outputs/smoke" \
  --device cuda:0 \
  --sizes 1b \
  --variants router_lora router_lora_expert_lora router_frozen \
             router_frozen_expert_lora uniform uniform_expert_lora \
             lowrank_router lowrank_router_expert_lora \
  --train-datasets wikitext piqa \
  --train-steps 2 --warmup-steps 1 \
  --batch-size 1 --micro-batch-size 1 --max-length 128 \
  --wikitext-eval-blocks 2 --piqa-eval-limit 16 \
  2>&1 | tee outputs/smoke.log
```

## 전체 실험

전체 grid는 3 sizes × (2 frozen baselines + 6 trainable variants × 2 train
datasets)입니다. 5B 조건까지 포함하므로 smoke 결과와 VRAM을 확인한 후
순차 실행하십시오.

```bash
"$PY" -u run_experiment.py \
  --phase all \
  --spread-root /home/xyzysh/work/SPREAD \
  --output-root "$PWD/outputs/olmoe_spread_lora" \
  --device cuda:0 \
  --sizes 1b 3b 5b \
  --train-datasets wikitext piqa \
  --train-steps 100 --warmup-steps 10 \
  --learning-rate 5e-5 \
  --rank 8 --alpha 8 \
  --batch-size 4 --micro-batch-size 1 --max-length 512 \
  --eval-tasks wikitext piqa \
  2>&1 | tee outputs/olmoe_spread_lora.log
```

중단 후 같은 명령을 다시 실행하면 완료된 `result.json`은 건너뛰고, 미완료
학습은 가장 최근 `checkpoints/step_*`에서 resume합니다. 결과는 다음에 생성됩니다.

```text
outputs/olmoe_spread_lora/
├── backbones/{1b,3b,5b}/
├── runs/<size>/<variant>/<train_dataset>/
│   ├── training.jsonl
│   ├── final/adapter.safetensors
│   └── result.json
└── reports/{results.json,results.csv,results.md}
```

이미 학습이 끝난 adapter만 재평가하려면:

```bash
"$PY" -u run_experiment.py \
  --phase evaluate \
  --spread-root /home/xyzysh/work/SPREAD \
  --output-root "$PWD/outputs/olmoe_spread_lora" \
  --device cuda:0
```

## 다른 서버로 이동

이 디렉터리는 `ICT-Express` 저장소에 포함되며, model checkpoint와 실행 결과가
실수로 올라가지 않도록 `outputs/`가 `.gitignore`되어 있습니다. 다른 서버에서는
이 저장소와 위에 명시한 SPREAD commit을 각각 clone한 뒤 `--spread-root`로
SPREAD 경로를 전달하십시오.
