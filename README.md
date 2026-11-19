# Absorb attention

Code for the paper "Absorb attention: An abstention gate for the first-token sink that selective attention
amplifies," submitted to Applied Soft Computing. It contains the model, the attention variants, the synthetic
tasks, the data preparation, training, evaluation and benchmark code of every experiment in the paper, and the
commands that run them.

## Installation

Python 3.11 and PyTorch with CUDA. The paper's compiled runs and zero-shot evaluations used PyTorch 2.14.1 with
CUDA 12.6 on NVIDIA A100 GPUs (the cu126 wheel below); the recorded environment was Python 3.11.11,
transformers 5.18.0, datasets 5.0.1, numpy 2.4.6 and lm-eval 0.4.13, the versions that `requirements.txt` pins.

```bash
git clone https://github.com/Parinyasan/absorb-attention
cd absorb-attention
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Every command below is run from the repository root.

## Data

```bash
python data/prepare_fineweb.py --tokens 2.1e9 --val_tokens 1e7 --out data/fineweb_edu
python data/prepare_fineweb.py --verify --out data/fineweb_edu
```

This streams FineWeb-Edu (sample-10BT) from the Hugging Face Hub and writes GPT-2 BPE tokens to
`data/fineweb_edu` (2.1B training tokens, 10M validation tokens). Training windows are drawn from the whole
training split, so keep these sizes to obtain the same data order. Set `HF_TOKEN` in the environment for
authenticated Hub access (optional). The PG-19 test books and the zero-shot datasets are downloaded by the
evaluation scripts.

## Models

Every model is a named preset in `presets.py`. The main models of the paper:

| Paper | Preset |
|---|---|
| Vanilla | `vanilla` |
| Vanilla + G1 | `gated` |
| SA | `selective` |
| Absorb (SA + G1) | `sa_g1` |
| Absorb-H | `sa_g1_head` |
| Absorb-N | `sa_null` |
| Absorb-N + delta write | `sa_null_delta` |
| Stack | `absorb_g1` |
| SA-wide | `selective_wide` |

## Training

`sweep.py` runs every job of a suite (one `train.py` process per job) on the GPUs given by `--gpus`, with
`--per_gpu` jobs per GPU. Finished runs are skipped, so a sweep can be restarted. `--dry` prints the commands.

```bash
python sweep.py --suite lm_main,lm_seeds,lm_confirm --gpus 0,1,2,3 --per_gpu 1
python sweep.py --suite lm_controls,lm_absorb,lm_factorial --gpus 0,1,2,3 --per_gpu 1
python sweep.py --suite lm_long_vanilla,lm_long_gated --gpus 0,1 --per_gpu 1
python sweep.py --suite lm50,lm50_gate --gpus 0,1,2,3 --per_gpu 1
python sweep.py --suite syn_mqar --gpus 0,1,2,3 --per_gpu 4
python sweep.py --suite syn_tasks --gpus 0,1,2,3 --per_gpu 4
python sweep.py --suite syn_ablate --gpus 0,1,2,3 --per_gpu 4
python sweep.py --suite syn_rate,syn_rate_long,syn_rate_tasks,syn_lr --gpus 0,1,2,3 --per_gpu 4
```

| Suite | Runs | Paper |
|---|---|---|
| `lm_main`, `lm_seeds`, `lm_confirm` | 124M models, 20k steps (1.31B tokens), seeds 0-2, eager with activation checkpointing | Sections 5.1-5.8, Tables 3-8, Figs. 2-4, Appendices B, C, E-J |
| `lm_controls` | 124M, compiled: seeds 3-4 of the SA decomposition, static nulls on SA and Vanilla, Vanilla + null slot, the residual-write and DDL controls, retirable first token, Absorb, Absorb-H and the Stack, and compiled reruns of SA and Absorb-N seed 0 | as above |
| `lm_absorb` | 124M, compiled: Absorb seeds 3-4, SA-wide seeds 0-2 | as above |
| `lm_factorial` | 124M, compiled: Vanilla and Vanilla + G1 seeds 3-4 | as above |
| `lm_long_vanilla`, `lm_long_gated` | Vanilla and Vanilla + G1 for 32k steps (2.1B tokens) | Sections 5.2, 5.5, 5.6, 5.8, Tables 3, 7, Appendices B, E, F, H |
| `lm50`, `lm50_gate` | 50M models, 8k steps | Section 5.9, Appendices D, E |
| `syn_mqar`, `syn_tasks`, `syn_ablate` | synthetic MQAR, variable tracking and needle tasks, component ablations | Appendices I, J |
| `syn_rate`, `syn_rate_long`, `syn_rate_tasks`, `syn_lr` | retirement-rate controls and the learning-rate sweep | Appendix I.4, I.5 |

`syn_lr` reuses the runs of `syn_tasks` at learning rate 1e-3, so run `syn_tasks` first.

## Evaluation

`scripts/ckpt_eval_all.py` evaluates every checkpoint under `runs/` that is not evaluated yet, one job per
checkpoint (for `--kind internals` the 50M suites are skipped unless named with `--suites` and
`--skip_suites ""`, as in the fourth command).

```bash
python scripts/ckpt_eval_all.py --kind zeroshot --lm_eval_version 0.4.13 --gpus 0,1,2,3 --per_gpu 2
python scripts/ckpt_eval_all.py --kind long --gpus 0,1,2,3 --per_gpu 1
python scripts/ckpt_eval_all.py --kind internals --what params,stats,window,quant,longdoc,rope,pos0,gate,gate2 --gpus 0,1,2,3 --per_gpu 1
python scripts/ckpt_eval_all.py --kind internals --what gate,gate2 --suites lm50,lm50_gate --skip_suites "" --gpus 0,1,2,3 --per_gpu 1
python scripts/ckpt_eval_all.py --kind internals --what evict,evict_matched --only sa_null_delta,selective,absorb_odds0,vanilla,sink,sa_constnull,sa_constnull0,selective_bos,sa_null_delta_bos,sa_g1,sa_g1_head,sa_null,sa_delta --gpus 0,1,2,3 --per_gpu 1
python scripts/ckpt_eval_all.py --kind internals --what evict,evict_matched --suites lm_long_gated --gpus 0,1,2,3 --per_gpu 1
mkdir -p results
python scripts/bench.py --model lm100m --seq_len 1024 --batch 8 --vocab 50304 --compile --variants vanilla,gated,selective,selective_wide,sa_null_delta,sa_g1_head,sa_g1,absorb_g1 --out results/bench_compile.json
```

| Command | Script | Paper |
|---|---|---|
| `--kind zeroshot` | `scripts/zeroshot.py`: eight zero-shot tasks with lm-evaluation-harness 0.4.13 | Table 8, Appendices D, H |
| `--kind long` | `scripts/long_context.py`: perplexity at 1-16 times the training length, passkey retrieval, KV-cache eviction at 1k | Tables 3, 4, 6, 7, Fig. 3, Appendices D, F, G, H, I |
| `--kind internals` (first two) | `scripts/internals_eval.py`: activation and sink statistics, learned null parameters, W8A8 quantization, position 0, sliding window and StreamingLLM, PG-19 long documents, RoPE scaling, abstention of the gate and the null slot | Tables 3-7, Appendices E, F, H, J |
| `--kind internals` (last two) | `scripts/internals_eval.py`: KV-cache eviction at 4k and 16k tokens | Section 5.6, Tables 3, 4, Fig. 4, Appendix G |
| `scripts/bench.py` | training throughput and memory, eager and compiled, at the 124M shape (one GPU) | Section 5.8, Table 9 |

`scripts/run_experiments.sh` runs all of the above in order (data, training, evaluation, tables); `GPUS`,
`PER_GPU` and `STEPS` select the GPUs, the synthetic jobs per GPU and the steps, for example
`GPUS=0,1,2,3 STEPS="lm124 long" bash scripts/run_experiments.sh`.

## Outputs

- `runs/<suite>/<run>/`: `results.json` (final metrics, validation history, per-layer statistics), `metrics.jsonl`
  (training log), `ckpt.pt` (language models), and `zeroshot.json`, `long_context.json`, `internals_eval.json`
  written by the evaluation; job logs in `runs/<suite>/_logs/`.
- `results/bench_compile.json`: the benchmark.
- `python aggregate.py <run dirs> --out <dir>` collects runs into one table per group (`all_runs.csv`,
  `summary.md`, eviction and per-layer tables, plots), for example

```bash
python aggregate.py runs/lm_main runs/lm_seeds runs/lm_confirm runs/lm_controls runs/lm_absorb runs/lm_factorial --out results/lm124
python aggregate.py runs/lm50 runs/lm50_gate --out results/lm50
python aggregate.py runs/syn_lr --out results/syn_lr
```

The 124M directories are read eager suites first: a later run with the same variant and seed (the two compiled
reruns in `lm_controls`) is reported separately and left out of the means.
