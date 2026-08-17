#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
icode_dir=$(cd "${repo_dir}/.." && pwd)

: "${MODEL_GGUF:?Set MODEL_GGUF to a Houmo Qwen GGUF containing M50 HMM assets}"

venv_dir=${FLASHRT_M50_VENV:-"${icode_dir}/.venv-m50-runtime"}
result_json=${RESULT_JSON:-"${repo_dir}/.cache/m50/results/qwen_tcim.json"}
prompt=${PROMPT:-请只回答一个数字：一加一等于多少？}
max_tokens=${MAX_TOKENS:-32}
repeat=${REPEAT:-3}
warmup=${WARMUP:-0}

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "Python environment not found: ${venv_dir}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_GGUF}" ]]; then
    echo "Model not found: ${MODEL_GGUF}" >&2
    exit 2
fi

mkdir -p "$(dirname "${result_json}")" "${repo_dir}/.cache/m50/xdg"

export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_DATA_HOME="${repo_dir}/.cache/m50/xdg"
export TCIM_BACKEND=Xh2HalBackend
export LD_LIBRARY_PATH="/opt/houmo-tcim-runtime-1.4.0/lib:/usr/local/houmo-sdk/hal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

args=(
    "${repo_dir}/examples/m50/qwen_tcim.py"
    --model "${MODEL_GGUF}"
    --prompt "${prompt}"
    --max-tokens "${max_tokens}"
    --repeat "${repeat}"
    --warmup "${warmup}"
    --output-json "${result_json}"
)
if [[ "${ZERO_KV_ON_RESET:-0}" == 1 ]]; then
    args+=(--zero-kv-on-reset)
fi

exec "${venv_dir}/bin/python" "${args[@]}"
