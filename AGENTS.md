# Repository Guidelines

`CLAUDE.md` is a symlink to this file. Update `AGENTS.md` first when shared
agent guidance changes.

## Path Index & Module Organization

Use this file as an index. When describing work, prefer the format
`surface: paths -> purpose -> validation`.

- `packaging: pyproject.toml, setup.py, ktransformers.py -> root shim that
  installs kt-kernel -> pip install .`
- `kernel-core: kt-kernel/python/, kt-kernel/operators/ -> bindings and native ops -> kt version plus focused pytest`
- `cpu-kernels: kt-kernel/cpu_backend/, kt-kernel/operators/{amx,avx2,llamafile,moe_kernel}/ -> AMX/AVX/llamafile paths -> run_suite.py --hw cpu --suite default`
- `cuda-kernels: kt-kernel/cuda/{fp8,gptq_marlin,moe,mxfp4}/ -> GPU kernels -> targeted CUDA pytest or model smoke`
- `inference: kt-kernel/python/cli/commands/{run,chat,model}.py,
  kt-kernel/python/utils/, kt-kernel/scripts/ -> serving and debug tools -> CLI/import smoke`
- `training: kt-kernel/python/sft/, doc/en/SFT/ -> KT-SFT and LoRA -> focused SFT tests`
- `tests: kt-kernel/test/, kt-kernel/test/per_commit/ -> pytest and CI coverage`
- `docs: README.md, kt-kernel/README.md, doc/en/, doc/zh/ -> user commands`

Treat `third_party/` as vendored code and `archive/` as legacy material unless
the task targets them.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: populate submodules.
- `./install.sh all --editable`: full editable install.
- `./install.sh kt-kernel --manual`: build only `kt-kernel` with explicit
  `CPUINFER_*` settings.
- `cd kt-kernel && ./install.sh build`: auto-detect CPU features.
- `cd kt-kernel && pip install -e .`: editable kernel install.
- `kt version`: verify the CLI and detected backend.

## Coding Style & Naming Conventions

Python uses Black with line length 120. Prefer type hints, snake_case, and
indexed modules. Native code follows `kt-kernel/.clang-format`, based on Google
style with a 120-column limit. Format native changes with:

```bash
cd kt-kernel
cmake -B build
cmake --build build --target format
```

## Testing Guidelines

Pytest is configured in `kt-kernel/pytest.ini`; tests and functions use
`test_*`. Markers include `cpu`, `cuda`, `amd`, `slow`, and `requires_model`.
For CI-style CPU validation:

```bash
cd kt-kernel/test
python3 run_suite.py --hw cpu --suite default
```

For focused work, run the smallest relevant target:
`cd kt-kernel && python -m pytest test/test_mxfp4_dequant.py -v`.

## Commit & Pull Request Guidelines

Use Conventional Commits where practical: `feat:`, `fix:`, `docs:`, `perf:`,
and `chore:`. Pull requests should name the indexed surface, list
verification, include benchmarks for performance changes, and update docs when
commands or model support change. The KT-Kernel workflow expects the `run-ci`
label and rejects draft PRs.

## Agent-Specific Notes

Do not commit generated build output, virtual environments, `.sindexer/`, or
`__pycache__/`. Keep changes scoped to the indexed surface unless the task
explicitly asks for cross-surface packaging, archive, or vendored changes.
