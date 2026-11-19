#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0}
PER_GPU=${PER_GPU:-4}
LM_EVAL_VERSION=${LM_EVAL_VERSION:-0.4.13}
STEPS=${STEPS:-"data syn syn_rate lm50 lm124 lm_long bench zeroshot long internals evict tables"}

LM124=lm_main,lm_seeds,lm_confirm,lm_controls,lm_absorb,lm_factorial
INTERNALS=params,stats,window,quant,longdoc,rope,pos0,gate,gate2
EVICT=sa_null_delta,selective,absorb_odds0,vanilla,sink,sa_constnull,sa_constnull0,selective_bos,sa_null_delta_bos,sa_g1,sa_g1_head,sa_null,sa_delta
BENCH=vanilla,gated,selective,selective_wide,sa_null_delta,sa_g1_head,sa_g1,absorb_g1

sweep() { python sweep.py --gpus "$GPUS" "$@"; }
evaluate() { python scripts/ckpt_eval_all.py --gpus "$GPUS" "$@"; }

for s in $STEPS; do
  echo "=== $s ($(date)) ==="
  case $s in
    data)
      if ! python data/prepare_fineweb.py --verify --out data/fineweb_edu; then
        python data/prepare_fineweb.py --tokens 2.1e9 --val_tokens 1e7 --out data/fineweb_edu || true
        python data/prepare_fineweb.py --verify --out data/fineweb_edu
      fi ;;
    syn)
      for x in syn_mqar syn_tasks syn_ablate; do sweep --suite "$x" --per_gpu "$PER_GPU"; done ;;
    syn_rate)
      sweep --suite syn_rate,syn_rate_long,syn_rate_tasks,syn_lr --per_gpu "$PER_GPU" ;;
    lm50)
      sweep --suite lm50,lm50_gate --per_gpu 1 ;;
    lm124)
      sweep --suite "$LM124" --per_gpu 1 ;;
    lm_long)
      sweep --suite lm_long_vanilla,lm_long_gated --per_gpu 1 ;;
    bench)
      mkdir -p results
      CUDA_VISIBLE_DEVICES=${GPUS%%,*} python scripts/bench.py --model lm100m --seq_len 1024 --batch 8 --vocab 50304 \
        --compile --variants "$BENCH" --out results/bench_compile.json ;;
    zeroshot)
      evaluate --kind zeroshot --per_gpu 2 --lm_eval_version "$LM_EVAL_VERSION" ;;
    long)
      evaluate --kind long --per_gpu 1 ;;
    internals)
      evaluate --kind internals --what "$INTERNALS" --per_gpu 1
      evaluate --kind internals --what gate,gate2 --suites lm50,lm50_gate --skip_suites "" --per_gpu 1 ;;
    evict)
      evaluate --kind internals --what evict,evict_matched --only "$EVICT" --per_gpu 1
      evaluate --kind internals --what evict,evict_matched --suites lm_long_gated --per_gpu 1 ;;
    tables)
      for x in syn_mqar syn_tasks syn_ablate syn_rate syn_rate_long syn_rate_tasks syn_lr lm_long_vanilla lm_long_gated; do
        python aggregate.py "runs/$x" --out "results/$x"
      done
      python aggregate.py runs/lm50 runs/lm50_gate --out results/lm50
      python aggregate.py runs/lm_main runs/lm_seeds runs/lm_confirm runs/lm_controls runs/lm_absorb runs/lm_factorial \
        --out results/lm124 ;;
    *)
      echo "unknown step $s"
      exit 1 ;;
  esac
done
