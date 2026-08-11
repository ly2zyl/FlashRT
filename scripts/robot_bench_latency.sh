#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run reproducible latency tests for robot policies.

Usage:
  scripts/robot_bench_latency.sh [all|pi05|smolvla_libero|groot_libero|vla_jepa_libero|molmoact2_libero] [OUTPUT_ROOT]

Defaults:
  model:       all
  output root: /workspace/models/robot-bench-results/latency

This script can be executed directly on the host. It starts the benchmark
container automatically. Every model uses the real seed-0 initial observation
from representative quick case libero_spatial:0.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/robot_bench_container.sh" run \
        scripts/robot_bench_latency.sh "$@"
fi

readonly TARGET="${1:-all}"
readonly OUTPUT_ROOT="${2:-/workspace/models/robot-bench-results/latency}"
readonly MODEL_ROOT="${ROBOT_BENCH_MODEL_ROOT:-/workspace/models}"
readonly REPRESENTATIVE_CASE_ARGS=(--libero-suite libero_spatial --libero-task-id 0 --seed 0)
export FLASH_RT_PALIGEMMA_TOKENIZER="${MODEL_ROOT}/paligemma_tokenizer.model"
export HF_HOME="${MODEL_ROOT}/hf-cache"
export TORCH_HOME="${MODEL_ROOT}/torch-cache"
export LIBERO_CONFIG_PATH="${MODEL_ROOT}/libero-config"
export MUJOCO_GL=egl

case "${TARGET}" in
    all|pi05|smolvla_libero|groot_libero|vla_jepa_libero|molmoact2_libero) ;;
    *)
        echo "unknown model: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

run_pi05() {
    echo "[latency] Pi0.5"
    /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py \
        --model pi05 \
        --checkpoint "${MODEL_ROOT}/pi05_libero_finetuned_v044" \
        --output "${OUTPUT_ROOT}/pi05/latency.json" \
        --warmup 10 --iters 30 "${REPRESENTATIVE_CASE_ARGS[@]}"
}

run_smolvla_libero() {
    echo "[latency] SmolVLA LIBERO"
    /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py \
        --model smolvla_libero \
        --checkpoint "${MODEL_ROOT}/SmolVLA-LIBERO" \
        --output "${OUTPUT_ROOT}/smolvla_libero/latency.json" \
        --warmup 10 --iters 30 "${REPRESENTATIVE_CASE_ARGS[@]}"
}

run_groot_libero() {
    echo "[latency] GR00T N1.7 LIBERO (spatial checkpoint)"
    /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py \
        --model groot_libero \
        --checkpoint "${MODEL_ROOT}/GR00T-N1.7-LIBERO/libero_spatial" \
        --output "${OUTPUT_ROOT}/groot_libero/latency.json" \
        --warmup 10 --iters 30 "${REPRESENTATIVE_CASE_ARGS[@]}"
}

run_vla_jepa_libero() {
    echo "[latency] VLA-JEPA LIBERO"
    /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py \
        --model vla_jepa_libero \
        --checkpoint "${MODEL_ROOT}/VLA-JEPA-LIBERO" \
        --output "${OUTPUT_ROOT}/vla_jepa_libero/latency.json" \
        --warmup 10 --iters 30 "${REPRESENTATIVE_CASE_ARGS[@]}"
}

run_molmoact2_libero() {
    echo "[latency] MolmoAct2 LIBERO"
    /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py \
        --model molmoact2_libero \
        --checkpoint "${MODEL_ROOT}/MolmoAct2-LIBERO-LeRobot" \
        --output "${OUTPUT_ROOT}/molmoact2_libero/latency.json" \
        --warmup 10 --iters 30 "${REPRESENTATIVE_CASE_ARGS[@]}"
}

case "${TARGET}" in
    all)
        run_pi05
        run_smolvla_libero
        run_groot_libero
        run_vla_jepa_libero
        run_molmoact2_libero
        ;;
    pi05) run_pi05 ;;
    smolvla_libero) run_smolvla_libero ;;
    groot_libero) run_groot_libero ;;
    vla_jepa_libero) run_vla_jepa_libero ;;
    molmoact2_libero) run_molmoact2_libero ;;
esac

echo "Latency results: ${OUTPUT_ROOT}"
