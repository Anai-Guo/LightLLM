#!/bin/sh
set -eu

# Keep knobs in lock-step with lightllm.sh so profile results line up 1:1.
export BS=${BS:-4}
export ILEN=${ILEN:-1000}
export OLEN=${OLEN:-10000}
export WARMUP_OLEN=${WARMUP_OLEN:-10}
export DECODE_PROFILE_STEPS=${DECODE_PROFILE_STEPS:-100}
export TP=${TP:-8}
export BENCH_MODE=${1:-${BENCH_MODE:-both}}
export CACHE_HIT_LEN=${CACHE_HIT_LEN:-$((ILEN - 1))}
# For static profiling, use the actual request total length by default rather
# than the online service ceiling; otherwise vLLM's throughput warmup compiles
# and profiles an unnecessarily huge sequence length and can OOM.
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((ILEN + OLEN))}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
export REASONING_PARSER=${REASONING_PARSER:-qwen3}
export LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-1}
export PERFORMANCE_MODE=${PERFORMANCE_MODE:-throughput}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-1}

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export ROOT_DIR
export MODEL_DIR=${MODEL_DIR:-../Qwen3.5-122B-A10B}
# Mirror lightllm's `./logs/forward_{prefill|decode}_<rank>/` layout. Only the
# top-level directory name differs (profiles/ vs logs/) so vLLM and lightllm
# artefacts don't clash when both are inspected side-by-side. Running
# BENCH_MODE=both will overwrite the static_forward traces with cache_hit ones,
# which matches lightllm.sh's behaviour.
export PROFILE_ROOT=${PROFILE_ROOT:-$ROOT_DIR/profiles}

# --- vLLM engine knobs ------------------------------------------------------
# Align attention backend with lightllm (flashinfer on both prefill & decode).
# export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
# Spawn start method for multiprocessing workers (see repo memory notes on
# vllm-benchmark spawn behaviour).
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# Profiler-related defaults.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export VLLM_ALLOW_INSECURE_SERIALIZATION=${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}

run_vllm_profile() {
  bench_mode=$1
  profile_stage=$2
  profile_dir=$PROFILE_ROOT

  mkdir -p "$profile_dir"

  echo "[vLLM] profiling ${bench_mode}/${profile_stage} -> ${profile_dir}"
  (
    cd "$ROOT_DIR"
    env -u PYTHONPATH \
      BENCH_MODE="$bench_mode" \
      PROFILE_STAGE="$profile_stage" \
      PROFILE_DIR="$profile_dir" \
      python "$ROOT_DIR/test/benchmark/static_inference/vllm_benchmark.py"
  )
}

run_static_forward_bench() {
  echo "[vLLM] run cold prefill+decode benchmark"
  run_vllm_profile static_forward full
}

run_cache_hit_forward_bench() {
  echo "[vLLM] run cache-hit prefill+decode benchmark, cache_hit_len=${CACHE_HIT_LEN}"
  run_vllm_profile cache_hit full
}

case "$BENCH_MODE" in
  static_forward)
    run_static_forward_bench
    ;;
  cache_hit)
    run_cache_hit_forward_bench
    ;;
  both)
    run_static_forward_bench
    run_cache_hit_forward_bench
    ;;
  *)
    echo "Unsupported BENCH_MODE: $BENCH_MODE" >&2
    echo "Supported values: static_forward, cache_hit, both" >&2
    exit 1
    ;;
esac
