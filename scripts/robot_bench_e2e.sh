#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run closed-loop end-to-end evaluation for all five LIBERO policies.

Usage:
  scripts/robot_bench_e2e.sh [quick|full] [OUTPUT_ROOT]

Protocols:
  quick  Six representative tasks, 10 episodes per model/task (300 total).
  full   Four suites, all 10 tasks, 10 episodes per model/task (2000 total).

Default output: /workspace/models/robot-bench-results/e2e
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f /.dockerenv ]]; then
    exec "${SCRIPT_DIR}/robot_bench_container.sh" run scripts/robot_bench_e2e.sh "$@"
fi

readonly PROTOCOL="${1:-quick}"
readonly OUTPUT_ROOT="${2:-/workspace/models/robot-bench-results/e2e}"
readonly MODEL_ROOT="${ROBOT_BENCH_MODEL_ROOT:-/workspace/models}"
export FLASH_RT_PALIGEMMA_TOKENIZER="${MODEL_ROOT}/paligemma_tokenizer.model"
export HF_HOME="${MODEL_ROOT}/hf-cache"
export TORCH_HOME="${MODEL_ROOT}/torch-cache"
export LIBERO_CONFIG_PATH="${MODEL_ROOT}/libero-config"
export MUJOCO_GL=egl

case "${PROTOCOL}" in
    quick)
        protocol_args=(
            --cases
            libero_spatial:0,libero_object:0,libero_goal:0,libero_goal:7,libero_10:0,libero_10:9
            --episodes 10
        )
        ;;
    full) protocol_args=(--task-ids all --episodes 10) ;;
    *)
        echo "unknown protocol: ${PROTOCOL}" >&2
        usage >&2
        exit 2
        ;;
esac

models=(pi05 smolvla_libero groot_libero vla_jepa_libero molmoact2_libero)
for model in "${models[@]}"; do
    echo "[e2e:${PROTOCOL}] ${model}"
    /opt/lerobot-venv/bin/python benchmarks/libero_e2e_eval.py \
        --model "${model}" \
        --model-root "${MODEL_ROOT}" \
        --output "${OUTPUT_ROOT}/${PROTOCOL}/${model}.json" \
        "${protocol_args[@]}"
done

echo "End-to-end results: ${OUTPUT_ROOT}/${PROTOCOL}"
