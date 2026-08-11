# FlashRT Repository Guide

## Scope and Sources of Truth

These instructions apply to the whole repository. Before changing code, read
the documentation closest to the task. Treat `README.md` and
`docker/README.md` as the build and deployment sources of truth,
`docs/stable_api.md` as the public Python contract, `docs/exec_contract.md` as
the execution-layer boundary, and `CONTRIBUTING.md` plus
`docs/pr_review_checklist.md` as the implementation and review standard.

Do not silently broaden a task to unrelated models, hardware families, or
generated artifacts. Preserve user changes in a dirty worktree.

## Repository Layout

- `flash_rt/`: Python API, hardware dispatch, frontends, model pipelines,
  calibration, weights, and runtime helpers.
- `csrc/`: CUDA kernels and pybind modules. Keep CMake source ownership and
  binding guards aligned.
- `runtime/`: stable model-runtime ABI and its native tests.
- `exec/`: model-agnostic `Buffer`/`Graph`/`Plan`/`Event` execution mechanism.
  Scheduling, sessions, protocol, and robot policy belong in `serving/`.
- `cpp/`: native C++ model/runtime integration and providers.
- `tests/`: main Python suite; component-local tests also live in
  `flash_rt/tests/`, `runtime/tests/`, `exec/tests/`, and `cpp/tests/`.
- `examples/`, `benchmarks/`, `serving/`, `training/`, and `docs/`: user flows,
  measurements, deployment hosts, training code, and design documentation.

## Docker-First Build and Deployment

For x86 NVIDIA systems, use the repository Docker image by default. It pins
the validated NGC CUDA/PyTorch/Python environment, clones CUTLASS 4.4.2 inside
the image, builds the extension modules, installs the package editable, and
runs an import smoke check during `docker build`.

Pin the target architecture instead of relying on build-time GPU discovery:

```bash
docker build -t flashrt:5090 \
  --build-arg GPU_ARCH=120 \
  -f docker/Dockerfile .

docker run --rm --gpus all flashrt:5090 \
  python3 -m pytest tests/test_install_smoke.py -q
```

Architecture values documented by the project are `120` for RTX 5090, `89`
for RTX 4090, `86` for RTX 3090, and `80` for A100. Use
`docker/Dockerfile.thor` and `--runtime=nvidia` on Jetson AGX Thor; do not use
the x86 image there. Optional image build arguments include `BASE_IMAGE`,
`CUTLASS_REF`, and `FA2_HDIMS`; only change them for an explicit deployment
requirement.

The Dockerfile's final import check is part of build validation. After the
image builds, run the focused pytest command above with GPU access. Broaden to
additional tests according to the changed area and available checkpoints.

## Native Build (Only When Requested)

Native builds require a supported Python 3.10–3.12 environment, CUDA toolkit,
matching PyTorch, pybind11, and CUTLASS 4.4.2 at
`third_party/cutlass`. Use an editable install because CMake writes `.so`
files directly into `flash_rt/`:

```bash
pip install -e ".[torch]"
cmake -B build -S . -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build -j$(nproc)
```

Remove `FA2_ARCH_NATIVE_ONLY=ON` for distributable multi-architecture
binaries. Never commit `.so` files, build directories, checkpoints, logs,
caches, or `third_party/cutlass`.

Standalone native layers can be configured and tested independently:

```bash
cmake -S runtime -B runtime/build -DBUILD_TESTING=ON
cmake --build runtime/build -j$(nproc)
ctest --test-dir runtime/build --output-on-failure
```

Use the equivalent component directory for `exec/` or `cpp/` when the change
is isolated there.

## Testing Expectations

Start with the smallest relevant tests, then expand by risk:

```bash
python -m pytest \
  tests/test_install_smoke.py \
  tests/test_load_model_use_fp8_kwarg.py \
  tests/test_calibration_helpers.py \
  -q
```

- Build/install changes: run `tests/test_install_smoke.py` and verify the
  compiled module path is under `flash_rt/`.
- Dispatch/API changes: run the relevant routing tests and update
  `docs/stable_api.md` for any public-contract change.
- Kernel, dtype, precision, calibration, or graph changes: run affected-model
  regression tests on the claimed GPU and compare against the appropriate
  reference path.
- Runtime/exec/C++ changes: run the component's CTest suite and Python gates.
- Tests with optional dependencies must skip cleanly with
  `pytest.importorskip(...)`.

Report tests that cannot run because checkpoints, fixtures, optional packages,
or hardware are unavailable. For GPU validation, record GPU model and compute
capability, driver and CUDA versions, framework version, `GPU_ARCH`, exact
commands, precision result, and latency metric. Distinguish end-to-end
wall-clock latency from CUDA Graph replay latency.

## Code and Architecture Rules

Use four-space indentation and `snake_case` for Python functions/modules,
`PascalCase` for classes, and C++17/CUDA 17 for native code. Match nearby style;
there is no repository-wide formatter, so avoid unrelated formatting.

Keep model and hardware routing explicit:

- model compute: `flash_rt/models/<model>/pipeline_<hw>.py`
- frontend: `flash_rt/frontends/<framework>/<model>_<hw>.py`
- dispatch: one `_PIPELINE_MAP` entry per supported
  `(config, framework, arch)` tuple

Do not add architecture branches inside shared pipelines. Keep
`flash_rt/hardware/<hw>/shared_primitives.py` model-agnostic.

Every unconditional pybind binding must have an implementation in every build
that exposes it. Hardware- or feature-gated kernels need the same guard in
CMake, source ownership, and bindings. Preserve legacy binding names, shapes,
dtypes, strictness, in-place behavior, and rounding contracts unless all
callers, tests, and docs change together. CUDA, cuBLASLt, CUTLASS, allocation,
and unsupported-shape failures must raise clear errors; do not return undefined
or zero-filled outputs after a failed required operation.

Keep `exec/` mechanism-only and independent of `csrc/`. Session state, cache
policy, scheduling, protocol handling, tools, and robot orchestration belong
in `serving/` or the host application.

## Commits and Reviews

Use short, imperative, technical commit subjects such as
`Fix FP8 descale GEMM error handling`. Keep a change focused on one behavior.
Update documentation for public API, build, hardware, checkpoint, or
performance changes. PR descriptions must list exact validation commands and
results, disclose unavailable fixtures/hardware, and include reproducible
precision or performance evidence for runtime changes.
