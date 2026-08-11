#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Create, start, enter, or run a command in the robot benchmark container.

Usage:
  scripts/robot_bench_container.sh enter          # default
  scripts/robot_bench_container.sh start
  scripts/robot_bench_container.sh status
  scripts/robot_bench_container.sh run COMMAND [ARG ...]

Environment overrides:
  ROBOT_BENCH_CONTAINER  Container name (default: flashrt-robot-bench)
  ROBOT_BENCH_IMAGE      Docker image (default: flashrt:5090)
  ROBOT_BENCH_ARTIFACTS  Host experiment archive directory
  ROBOT_BENCH_HF_XET_CONCURRENCY  Large-weight download concurrency (default: 64)
EOF
}

readonly ACTION="${1:-enter}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
    usage
    exit 0
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly CONTAINER="${ROBOT_BENCH_CONTAINER:-flashrt-robot-bench}"
readonly IMAGE="${ROBOT_BENCH_IMAGE:-flashrt:5090}"
readonly ARTIFACTS="${ROBOT_BENCH_ARTIFACTS:-${REPO_ROOT}-profile-artifacts-20260804}"
readonly MODEL_VOLUME="flashrt-pi05-models"
readonly HF_XET_CONCURRENCY="${ROBOT_BENCH_HF_XET_CONCURRENCY:-64}"
readonly TOKENIZER_PATH="/workspace/models/paligemma_tokenizer.model"
readonly TOKENIZER_URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
readonly LEROBOT_PYTHON="/opt/lerobot-venv/bin/python"
readonly LIBERO_ASSETS_DIR="/workspace/models/libero-assets"
readonly LIBERO_ASSETS_REVISION="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
readonly SMOLVLA_MODEL_DIR="/workspace/models/SmolVLA-LIBERO"
readonly SMOLVLA_REPO="lerobot/smolvla_libero"
readonly SMOLVLA_REVISION="31d453f7edd78c839a8bbc39744a292686daf0de"
readonly GROOT_BASE_DIR="/workspace/models/GR00T-N1.7-3B"
readonly GROOT_BASE_REPO="nvidia/GR00T-N1.7-3B"
readonly GROOT_BASE_REVISION="2fc962b973bccdd5d8ce4f67cc63b264d6886495"
readonly GROOT_BASE_SHARD_1_SHA256="8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d"
readonly GROOT_BASE_SHARD_2_SHA256="c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661"
readonly GROOT_LIBERO_DIR="/workspace/models/GR00T-N1.7-LIBERO"
readonly GROOT_SPATIAL_REPO="nvidia/gr00t17-lerobot-libero_spatial-640"
readonly GROOT_SPATIAL_REVISION="32a6ec786d6509df31b40392b4e4dcdda78c0f11"
readonly GROOT_SPATIAL_SHA256="8a6e55ca705ab60c6d9c1eaf585dbda53f1b92bffdf6050c18ce68eb3afc6e69"
readonly GROOT_OBJECT_REPO="nvidia/gr00t17-lerobot-libero_object-640"
readonly GROOT_OBJECT_REVISION="1499db357f6ca3762b56c2e8c00b530eb9a09444"
readonly GROOT_OBJECT_SHA256="8ce64a77ae445f0647fc33fb96660ec01ac97ef3c9b351249b1fc75422705cc9"
readonly GROOT_GOAL_REPO="nvidia/gr00t17-lerobot-libero_goal-640"
readonly GROOT_GOAL_REVISION="436c57c0eb7a90be54270abf3977668b2084ad75"
readonly GROOT_GOAL_SHA256="71c8220d03c1d429fe90861869e82189e944aeef9a58aea13bc4f3140dfdb0ef"
readonly GROOT_10_REPO="nvidia/gr00t17-lerobot-libero_10-640"
readonly GROOT_10_REVISION="5ee08ab09fac5c5ef2388a14c882ea825ac861db"
readonly GROOT_10_SHA256="8a7f3c0fb13cc84f89bbc7af1a675431ddd7b16ce47bebb12b4298f6b2827a94"
readonly VLA_JEPA_MODEL_DIR="/workspace/models/VLA-JEPA-LIBERO"
readonly VLA_JEPA_REPO="lerobot/VLA-JEPA-LIBERO"
readonly VLA_JEPA_REVISION="735d9f692981e286ade093b5046627eda876e5d0"
readonly VLA_JEPA_SHA256="b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c"
readonly VLA_JEPA_QWEN_DIR="/workspace/models/VLA-JEPA-deps/Qwen3-VL-2B-Instruct"
readonly VLA_JEPA_QWEN_REPO="Qwen/Qwen3-VL-2B-Instruct"
readonly VLA_JEPA_QWEN_REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
readonly VLA_JEPA_QWEN_SHA256="7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
readonly VLA_JEPA_ENCODER_DIR="/workspace/models/VLA-JEPA-deps/vjepa2-vitl-fpc64-256"
readonly VLA_JEPA_ENCODER_REPO="facebook/vjepa2-vitl-fpc64-256"
readonly VLA_JEPA_ENCODER_REVISION="b3c1679b7c34d3255ef3547f27c7b226aefab26f"
readonly VLA_JEPA_ENCODER_SHA256="25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b"
readonly MOLMOACT2_MODEL_DIR="/workspace/models/MolmoAct2-LIBERO-LeRobot"
readonly MOLMOACT2_REPO="allenai/MolmoAct2-LIBERO-LeRobot"
readonly MOLMOACT2_REVISION="f0c5a9567c2b72faadec901e16e055c8b098c2f5"
readonly MOLMOACT2_SHA256="4bc4cae246588dddfb558c3e082db4a1e0e21ac25b3aefc5e4d91068b845e8cf"
readonly MOLMOACT2_BASE_DIR="/workspace/models/MolmoAct2-LIBERO-Base"
readonly MOLMOACT2_BASE_REPO="allenai/MolmoAct2-LIBERO"
readonly MOLMOACT2_BASE_REVISION="0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
readonly MOLMOACT2_TOKENIZER_DIR="/workspace/models/MolmoAct2-FAST-Tokenizer"
readonly MOLMOACT2_TOKENIZER_REPO="allenai/MolmoAct2-FAST-Tokenizer"
readonly MOLMOACT2_TOKENIZER_REVISION="d45593b4c863d0bc1ca064f8b352fa16b75c38e8"
readonly SMOLVLM_MODEL_DIR="/workspace/models/SmolVLM2-500M-Video-Instruct"
readonly SMOLVLM_REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
readonly SMOLVLM_REVISION="7b375e1b73b11138ff12fe22c8f2822d8fe03467"

require_host_tools() {
    command -v docker >/dev/null 2>&1 || {
        echo "docker is not installed or not on PATH" >&2
        exit 1
    }
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || {
        echo "Docker image not found: ${IMAGE}" >&2
        exit 1
    }
}

verify_model_sha256() {
    local label="$1"
    local model_path="$2"
    local expected="$3"
    local actual
    actual="$(
        docker exec "${CONTAINER}" sha256sum "${model_path}" | awk '{print $1}'
    )"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "${label} checkpoint SHA256 mismatch: ${actual}" >&2
        exit 1
    fi
}

hf_download_with_retry() {
    local attempt=1
    while (( attempt <= 20 )); do
        if docker exec \
            --env HTTP_PROXY --env HTTPS_PROXY --env ALL_PROXY \
            --env HF_ENDPOINT=https://huggingface.co \
            --env HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false \
            --env "HF_XET_FIXED_DOWNLOAD_CONCURRENCY=${HF_XET_CONCURRENCY}" \
            "${CONTAINER}" hf download "$@"; then
            return 0
        fi
        echo "Hugging Face download interrupted; retry ${attempt}/20" >&2
        ((attempt += 1))
        sleep 3
    done
    echo "Hugging Face download failed after 20 attempts" >&2
    return 1
}

download_groot_libero_checkpoint() {
    local suite="$1"
    local repo="$2"
    local revision="$3"
    local sha256="$4"
    local destination="${GROOT_LIBERO_DIR}/${suite}"
    if ! docker exec "${CONTAINER}" test -s \
        "${destination}/model.safetensors"; then
        echo "Downloading GR00T N1.7 ${suite} checkpoint (first run only)"
        hf_download_with_retry "${repo}" \
            --revision "${revision}" --local-dir "${destination}" \
            --max-workers 4
    fi
    verify_model_sha256 "GR00T-N1.7-${suite}" \
        "${destination}/model.safetensors" "${sha256}"
}

resolve_extension() {
    local module="$1"
    local matches=("${ARTIFACTS}/build/${module}"*.so)
    if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
        echo "expected one ${module}*.so under ${ARTIFACTS}/build" >&2
        exit 1
    fi
    printf '%s\n' "${matches[0]}"
}

create_container() {
    resolve_extension flash_rt_kernels >/dev/null
    resolve_extension flash_rt_fa2 >/dev/null

    docker volume inspect "${MODEL_VOLUME}" >/dev/null 2>&1 || \
        docker volume create "${MODEL_VOLUME}" >/dev/null

    docker create \
        --name "${CONTAINER}" \
        --init \
        --gpus all \
        --ipc=host \
        --network=host \
        --cap-add SYS_ADMIN \
        --shm-size=8g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --env HTTP_PROXY= \
        --env HTTPS_PROXY= \
        --env ALL_PROXY= \
        --env "FLASH_RT_PALIGEMMA_TOKENIZER=${TOKENIZER_PATH}" \
        --env FLASHRT_EXTENSION_DIR=/opt/flashrt-extensions \
        --env HF_HOME=/workspace/models/hf-cache \
        --env TORCH_HOME=/workspace/models/torch-cache \
        --env LIBERO_CONFIG_PATH=/workspace/models/libero-config \
        --env MUJOCO_GL=egl \
        --workdir /workspace/FlashRT \
        --mount "type=bind,src=${REPO_ROOT},dst=/workspace/FlashRT" \
        --mount "type=volume,src=${MODEL_VOLUME},dst=/workspace/models" \
        --mount "type=bind,src=${ARTIFACTS}/build,dst=/opt/flashrt-extensions,readonly" \
        "${IMAGE}" sleep infinity >/dev/null
    echo "Created ${CONTAINER}"
}

start_container() {
    if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
        create_container
    fi
    if [[ "$(docker container inspect --format '{{.State.Running}}' \
        "${CONTAINER}")" != "true" ]]; then
        docker start "${CONTAINER}" >/dev/null
        echo "Started ${CONTAINER}"
    fi

    if ! docker exec "${CONTAINER}" test -f /opt/flashrt-libero-egl-ready-v1; then
        echo "Installing the LIBERO headless EGL runtime (first run only)"
        docker exec "${CONTAINER}" bash -lc \
            'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libegl1 libopengl0 libgl1'
        docker exec "${CONTAINER}" touch /opt/flashrt-libero-egl-ready-v1
    fi

    if ! docker exec "${CONTAINER}" test -s "${TOKENIZER_PATH}"; then
        echo "Downloading the shared PaliGemma tokenizer (first run only)"
        docker exec "${CONTAINER}" curl -fL "${TOKENIZER_URL}" \
            -o "${TOKENIZER_PATH}.partial"
        docker exec "${CONTAINER}" mv \
            "${TOKENIZER_PATH}.partial" "${TOKENIZER_PATH}"
    fi
    if ! docker exec "${CONTAINER}" test -x "${LEROBOT_PYTHON}"; then
        echo "Creating the isolated LeRobot runtime (first run only)"
        docker exec "${CONTAINER}" python3 -m venv \
            --system-site-packages /opt/lerobot-venv
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip install \
            --no-cache-dir lerobot==0.6.1 transformers==5.14.1
    fi
    if ! docker exec "${CONTAINER}" test -f /opt/flashrt-libero-vla-ready-v1; then
        echo "Preparing the LIBERO VLA runtime (first run only)"
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip install \
            --no-cache-dir num2words==0.5.14
        docker exec "${CONTAINER}" touch /opt/flashrt-libero-vla-ready-v1
    fi
    if ! docker exec "${CONTAINER}" test -f /opt/flashrt-groot-n17-ready-v1; then
        echo "Preparing the GR00T N1.7 inference runtime (first run only)"
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip install \
            --no-cache-dir dm-tree==0.1.9 qwen-vl-utils==0.0.14
        docker exec "${CONTAINER}" touch /opt/flashrt-groot-n17-ready-v1
    fi
    if ! docker exec "${CONTAINER}" test -f /opt/flashrt-libero-e2e-ready-v1; then
        echo "Preparing the LIBERO closed-loop evaluation environment (first run only)"
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip install \
            --no-cache-dir datasets==4.8.5 pandas==2.3.3 \
            jsonlines==4.0.0 av==15.1.0 hf-libero==0.1.4 \
            transformers==5.14.1
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip uninstall \
            --yes opencv-python
        docker exec "${CONTAINER}" "${LEROBOT_PYTHON}" -m pip install \
            --no-cache-dir --force-reinstall --no-deps \
            opencv-python-headless==4.13.0.92
        docker exec "${CONTAINER}" touch /opt/flashrt-libero-e2e-ready-v1
    fi
    if ! docker exec "${CONTAINER}" test -f \
        /workspace/models/libero-config/config.yaml; then
        echo "Initializing the persistent LIBERO environment paths"
        docker exec \
            --env LIBERO_CONFIG_PATH=/workspace/models/libero-config \
            "${CONTAINER}" bash -lc \
            'printf "n\n" | /opt/lerobot-venv/bin/python -c "import libero.libero"'
    fi
    if ! docker exec "${CONTAINER}" test -s \
        "${LIBERO_ASSETS_DIR}/scenes/libero_tabletop_base_style.xml"; then
        echo "Downloading the pinned LIBERO simulation assets (first run only)"
        docker exec \
            --env HTTP_PROXY --env HTTPS_PROXY --env ALL_PROXY \
            --env HF_ENDPOINT=https://huggingface.co \
            --env HF_HUB_DISABLE_XET=1 \
            "${CONTAINER}" hf download lerobot/libero-assets \
            --repo-type dataset --revision "${LIBERO_ASSETS_REVISION}" \
            --local-dir "${LIBERO_ASSETS_DIR}" --max-workers 2
    fi
    docker exec "${CONTAINER}" ln -sfn "${LIBERO_ASSETS_DIR}" \
        /opt/lerobot-venv/lib/python3.12/site-packages/libero/libero/assets
    if ! docker exec "${CONTAINER}" test -s \
        "${SMOLVLA_MODEL_DIR}/model.safetensors"; then
        echo "Downloading SmolVLA LIBERO checkpoint (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            --env HF_HUB_DISABLE_XET=1 \
            "${CONTAINER}" hf download "${SMOLVLA_REPO}" \
            --revision "${SMOLVLA_REVISION}" \
            --include config.json model.safetensors README.md \
                'policy_*.json' 'policy_*.safetensors' train_config.json \
            --local-dir "${SMOLVLA_MODEL_DIR}"
    fi
    if ! docker exec "${CONTAINER}" test -s \
        "${GROOT_BASE_DIR}/model-00002-of-00002.safetensors"; then
        echo "Downloading the pinned GR00T N1.7 base model (first run only)"
        hf_download_with_retry "${GROOT_BASE_REPO}" \
            --revision "${GROOT_BASE_REVISION}" \
            --local-dir "${GROOT_BASE_DIR}" --max-workers 4
    fi
    verify_model_sha256 GR00T-N1.7-base-shard-1 \
        "${GROOT_BASE_DIR}/model-00001-of-00002.safetensors" \
        "${GROOT_BASE_SHARD_1_SHA256}"
    verify_model_sha256 GR00T-N1.7-base-shard-2 \
        "${GROOT_BASE_DIR}/model-00002-of-00002.safetensors" \
        "${GROOT_BASE_SHARD_2_SHA256}"
    download_groot_libero_checkpoint libero_spatial \
        "${GROOT_SPATIAL_REPO}" "${GROOT_SPATIAL_REVISION}" \
        "${GROOT_SPATIAL_SHA256}"
    download_groot_libero_checkpoint libero_object \
        "${GROOT_OBJECT_REPO}" "${GROOT_OBJECT_REVISION}" \
        "${GROOT_OBJECT_SHA256}"
    download_groot_libero_checkpoint libero_goal \
        "${GROOT_GOAL_REPO}" "${GROOT_GOAL_REVISION}" \
        "${GROOT_GOAL_SHA256}"
    download_groot_libero_checkpoint libero_10 \
        "${GROOT_10_REPO}" "${GROOT_10_REVISION}" \
        "${GROOT_10_SHA256}"
    if ! docker exec "${CONTAINER}" test -s \
        "${VLA_JEPA_MODEL_DIR}/model.safetensors"; then
        echo "Downloading VLA-JEPA LIBERO checkpoint (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${VLA_JEPA_REPO}" \
            --revision "${VLA_JEPA_REVISION}" \
            --local-dir "${VLA_JEPA_MODEL_DIR}"
    fi
    verify_model_sha256 VLA-JEPA \
        "${VLA_JEPA_MODEL_DIR}/model.safetensors" "${VLA_JEPA_SHA256}"
    if ! docker exec "${CONTAINER}" test -s \
        "${VLA_JEPA_QWEN_DIR}/model.safetensors"; then
        echo "Downloading the VLA-JEPA vision-language backbone (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${VLA_JEPA_QWEN_REPO}" \
            --revision "${VLA_JEPA_QWEN_REVISION}" \
            --local-dir "${VLA_JEPA_QWEN_DIR}"
    fi
    verify_model_sha256 VLA-JEPA-Qwen-backbone \
        "${VLA_JEPA_QWEN_DIR}/model.safetensors" "${VLA_JEPA_QWEN_SHA256}"
    if ! docker exec "${CONTAINER}" test -s \
        "${VLA_JEPA_ENCODER_DIR}/model.safetensors"; then
        echo "Downloading the VLA-JEPA video encoder (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${VLA_JEPA_ENCODER_REPO}" \
            --revision "${VLA_JEPA_ENCODER_REVISION}" \
            --exclude 'original/*' \
            --local-dir "${VLA_JEPA_ENCODER_DIR}"
    fi
    verify_model_sha256 VLA-JEPA-video-encoder \
        "${VLA_JEPA_ENCODER_DIR}/model.safetensors" \
        "${VLA_JEPA_ENCODER_SHA256}"
    if ! docker exec "${CONTAINER}" test -s \
        "${MOLMOACT2_BASE_DIR}/model-00005-of-00005.safetensors"; then
        echo "Downloading the MolmoAct2 LIBERO backbone (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${MOLMOACT2_BASE_REPO}" \
            --revision "${MOLMOACT2_BASE_REVISION}" \
            --local-dir "${MOLMOACT2_BASE_DIR}"
    fi
    if ! docker exec "${CONTAINER}" test -s \
        "${MOLMOACT2_MODEL_DIR}/model.safetensors"; then
        echo "Downloading MolmoAct2 LIBERO LeRobot checkpoint (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${MOLMOACT2_REPO}" \
            --revision "${MOLMOACT2_REVISION}" \
            --local-dir "${MOLMOACT2_MODEL_DIR}"
    fi
    if ! docker exec "${CONTAINER}" test -s \
        "${MOLMOACT2_TOKENIZER_DIR}/tokenizer.json"; then
        echo "Downloading the MolmoAct2 action tokenizer (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            "${CONTAINER}" hf download "${MOLMOACT2_TOKENIZER_REPO}" \
            --revision "${MOLMOACT2_TOKENIZER_REVISION}" \
            --local-dir "${MOLMOACT2_TOKENIZER_DIR}"
    fi
    verify_model_sha256 MolmoAct2 \
        "${MOLMOACT2_MODEL_DIR}/model.safetensors" "${MOLMOACT2_SHA256}"
    if ! docker exec "${CONTAINER}" test -s \
        "${SMOLVLM_MODEL_DIR}/model.safetensors"; then
        echo "Downloading the SmolVLA vision-language backbone (first run only)"
        docker exec \
            --env HF_ENDPOINT=https://hf-mirror.com \
            --env HF_HUB_DISABLE_XET=1 \
            "${CONTAINER}" hf download "${SMOLVLM_REPO}" \
            --revision "${SMOLVLM_REVISION}" \
            --exclude 'onnx/*' \
            --local-dir "${SMOLVLM_MODEL_DIR}"
    fi
}

require_host_tools

case "${ACTION}" in
    start)
        start_container
        echo "Container ready: ${CONTAINER}"
        ;;
    status)
        if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
            docker container inspect --format \
                'name={{.Name}} image={{.Config.Image}} status={{.State.Status}}' \
                "${CONTAINER}"
        else
            echo "Container does not exist: ${CONTAINER}"
        fi
        ;;
    enter)
        start_container
        exec docker exec --interactive --tty \
            --workdir /workspace/FlashRT "${CONTAINER}" bash
        ;;
    run)
        if [[ $# -eq 0 ]]; then
            echo "run requires a command" >&2
            usage >&2
            exit 2
        fi
        start_container
        exec docker exec --interactive \
            --workdir /workspace/FlashRT "${CONTAINER}" "$@"
        ;;
    *)
        echo "unknown action: ${ACTION}" >&2
        usage >&2
        exit 2
        ;;
esac
