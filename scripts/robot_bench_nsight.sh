#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run Nsight Systems and Nsight Compute for all benchmark models.

Usage:
  scripts/robot_bench_nsight.sh [MODEL] [OUTPUT_ROOT] [full|systems]

Models:
  all, pi05, smolvla_libero, groot_libero, vla_jepa_libero, molmoact2_libero

Defaults:
  model:       all
  output root: /workspace/models/robot-bench-results/nsight
  mode:        full (Nsight Systems + Nsight Compute)

This script can be executed directly on the host. It starts the benchmark
container automatically. Every model uses the real seed-0 initial observation
from quick case libero_spatial:0. NCU profiles one representative hot kernel.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/robot_bench_container.sh" run \
        scripts/robot_bench_nsight.sh "$@"
fi

readonly TARGET="${1:-all}"
readonly OUTPUT_ROOT="${2:-/workspace/models/robot-bench-results/nsight}"
readonly PROFILE_MODE="${3:-full}"
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
case "${PROFILE_MODE}" in
    full|systems) ;;
    *)
        echo "unknown profile mode: ${PROFILE_MODE}" >&2
        usage >&2
        exit 2
        ;;
esac

require_tools() {
    local required=(python3 nsys)
    if [[ "${PROFILE_MODE}" == "full" ]]; then
        required+=(ncu)
    fi
    for command_name in "${required[@]}"; do
        command -v "${command_name}" >/dev/null 2>&1 || {
            echo "required command not found: ${command_name}" >&2
            exit 1
        }
    done
}

export_nsys_stats() {
    local report="$1"
    local output_prefix="$2"
    nsys stats \
        --report cuda_gpu_kern_sum,cuda_api_sum \
        --format csv \
        --output "${output_prefix}" \
        --force-overwrite=true --force-export=true \
        "${report}"
}

export_ncu_details() {
    local report="$1"
    local output_csv="$2"
    ncu --import "${report}" --page details --csv > "${output_csv}"
}

profile_libero_policy() {
    local model="$1"
    local checkpoint="$2"
    local kernel_regex="$3"
    local model_dir="${OUTPUT_ROOT}/${model}"
    local benchmark=(
        /opt/lerobot-venv/bin/python benchmarks/libero_policy_profile.py
        --model "${model}"
        --checkpoint "${checkpoint}"
        "${REPRESENTATIVE_CASE_ARGS[@]}"
    )
    mkdir -p "${model_dir}"

    echo "[nsys] ${model}"
    nsys profile \
        --trace=cuda,nvtx,osrt,cublas,cudnn \
        --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --cuda-graph-trace=node --force-overwrite=true \
        --output "${model_dir}/nsys" \
        "${benchmark[@]}" \
        --warmup 3 --iters 0 --profile-iters 1 \
        --output "${model_dir}/nsys_run.json"
    export_nsys_stats \
        "${model_dir}/nsys.nsys-rep" "${model_dir}/nsys_stats"

    if [[ "${PROFILE_MODE}" == "systems" ]]; then
        return
    fi

    echo "[ncu] ${model}"
    ncu \
        --profile-from-start off --target-processes all \
        --replay-mode kernel --kernel-name-base demangled \
        --kernel-name "${kernel_regex}" --launch-count 1 \
        --set full --force-overwrite --export "${model_dir}/ncu" \
        "${benchmark[@]}" \
        --warmup 3 --iters 0 --profile-iters 1 \
        --output "${model_dir}/ncu_run.json"
    export_ncu_details \
        "${model_dir}/ncu.ncu-rep" "${model_dir}/ncu_details.csv"
}

run_pi05() {
    profile_libero_policy \
        pi05 "${MODEL_ROOT}/pi05_libero_finetuned_v044" \
        'regex:.*qqtst_mma_128x128x64.*'
}

run_smolvla_libero() {
    profile_libero_policy smolvla_libero "${MODEL_ROOT}/SmolVLA-LIBERO" \
        'regex:.*(gemm|mm|convolve).*'
}

run_groot_libero() {
    profile_libero_policy groot_libero \
        "${MODEL_ROOT}/GR00T-N1.7-LIBERO/libero_spatial" \
        'regex:.*(gemm|mm|convolve).*'
}

run_vla_jepa_libero() {
    profile_libero_policy vla_jepa_libero "${MODEL_ROOT}/VLA-JEPA-LIBERO" \
        'regex:.*(gemm|mm|convolve).*'
}

run_molmoact2_libero() {
    profile_libero_policy molmoact2_libero \
        "${MODEL_ROOT}/MolmoAct2-LIBERO-LeRobot" \
        'regex:.*(gemm|mm|convolve).*'
}

require_tools
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

echo "Nsight results: ${OUTPUT_ROOT}"
