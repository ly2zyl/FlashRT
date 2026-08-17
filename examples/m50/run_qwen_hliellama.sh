#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
icode_dir=$(cd "${repo_dir}/.." && pwd)

: "${MODEL_GGUF:?Set MODEL_GGUF to a Houmo Qwen GGUF containing M50 HMM artifacts}"

venv_dir=${FLASHRT_M50_VENV:-"${icode_dir}/.venv-m50-runtime"}
houmo_llama_root=${HOUMO_LLAMA_ROOT:-/home/sky/houmo-HLIELLama-xh2}
provider_lib=${FLASHRT_PROVIDER_LIB:-"${repo_dir}/build/houmo-llama/libflashrt_cpp_llama_cpp_provider_c.so"}
result_json=${RESULT_JSON:-"${repo_dir}/artifacts/m50_qwen/results/qwen_flashrt_run.json"}
mode=${MODE:-staged}
prompt=${PROMPT:-请只回答一个数字：一加一等于多少？}
max_tokens=${MAX_TOKENS:-32}
ctx_size=${CTX_SIZE:-512}
threads=${THREADS:-8}
repeat=${REPEAT:-1}

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "Python environment not found: ${venv_dir}" >&2
    exit 2
fi
if [[ ! -f "${provider_lib}" ]]; then
    echo "FlashRT provider not found: ${provider_lib}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_GGUF}" ]]; then
    echo "Model not found: ${MODEL_GGUF}" >&2
    exit 2
fi

mkdir -p "$(dirname "${result_json}")" "${icode_dir}/.local-share"

export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export XDG_DATA_HOME="${icode_dir}/.local-share"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:${repo_dir}/build/houmo-llama/runtime:${houmo_llama_root}/lib:/opt/houmo-tcim-runtime-1.4.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

args=(
    "${repo_dir}/examples/m50/qwen_hliellama.py"
    --model "${MODEL_GGUF}"
    --provider-lib "${provider_lib}"
    --mode "${mode}"
    --prompt "${prompt}"
    --max-tokens "${max_tokens}"
    --ctx-size "${ctx_size}"
    --threads "${threads}"
    --repeat "${repeat}"
    --output-json "${result_json}"
)
if [[ -n "${MODEL_SHA256_FILE:-}" ]]; then
    args+=(--model-sha256-file "${MODEL_SHA256_FILE}")
fi

exec "${venv_dir}/bin/python" "${args[@]}"
