# AGENTS.md

Bachelor's thesis GP-benchmark project (repo `JunE1404/GP-Benchmark`, branch `main`). Benchmarks ExactGP / ExactGP-CG / SVGP / CAGP on synthetic + UCI regression data.

## Hard rule

**Never edit anything in this project (files, configs, deps, site-packages) unless the user explicitly asks.** If an instruction is ambiguous, ask first. This includes "obvious" fixes.

## Environment

- **Run everything in WSL2 Ubuntu at `/home/gp-project`** (a full copy of this repo incl. `.git`). Native Windows is broken: pykeops 2.3 hard-imports Unix-only `fcntl` at `pykeops/common/utils.py:1`, crashing `import gpytorch`.
- Working copy + Windows checkout can diverge; they already do (see deps). Never assume they're in sync.

## Commands (run in WSL: `wsl -d Ubuntu -e bash -lc "..."`)

```bash
cd /home/gp-project && uv sync
cd /home/gp-project && uv run python src/main.py --config run_configs/test.json
```

No tests, lint, or typecheck commands exist. `[tool.basedpyright]` settings are lenient.

## Dependency pins (fragile, verified, do not "fix")

- `gpytorch` from git branch **`computation-aware-gps-v3`** — required. The `computation-aware-gps-v2` branch's `ComputationAwareELBO` calls the private kernel method `_forward_no_kernel_linop`, which only `MaternKernel` implements → `AttributeError` with `RBF`/`RBFKeops`; v3 fixed it by calling the kernel's normal `forward`.
- `linear-operator` from git branch **`sparsity`** (pins `jaxtyping==0.2.19`).
- `[tool.uv] override-dependencies = ["jaxtyping>=0.2.20"]` — required; without it the pin conflicts with `gpjax>=0.14.0` → `equinox>=0.11.0`.
- `torch` from explicit `pytorch-cu130` index; Python 3.14 (`.python-version`), `requires-python >=3.13`.
- **Known divergence**: Windows `pyproject.toml` still pins gpytorch v2; WSL copy pins v3. Do not reconcile without explicit instruction.

## How runs work

- Entrypoint `src/main.py`; configs in `run_configs/*.json` (fields: `dataset`, `data_split`, `data_standartization`, `device`, `gp`, `kernel`, `likelyhood`, `mean`, `optimizer`, `learningrate`, `iterations`, `approximation_size`, `seed`, `shuffle`) or CLI flags.
- `gp` options: `exact`, `exactcg`, `svgp`, `cagp` (`approximation_size` = projection dim/actions for CAGP, inducing points for SVGP). Kernel options: `RBF`, `matern2.5`, `RBFKeops`, `matern2.5Keops` — all wrapped in `ScaleKernel` in `main.py`'s `run()`.
- **Results are saved CWD-relative**: `results/<dataset>/<model>/<timestamp>.json` (`src/main.py:379`). Running from project root → `results/` at root; from `src/` → `src/results/` (gitignored). Same applies to dataset cache `datasets/localfiles/`.
- Results JSON records repo HEAD via `helpers.get_git_revision_hash()`. `helpers.check_repo_clean()` exists but is commented out in `main.py`.

## GPU / KeOps in WSL

- `torch.cuda.is_available()` works in WSL. KeOps GPU kernels additionally need `nvcc` on PATH (pip `cuda-toolkit` installs it inside `.venv/.../nvidia/cuda_toolkit/bin`, not on PATH); without it KeOps falls back to CPU-only with a warning. Torch itself is unaffected.

## Misc

- `src/notes.txt` contains the thesis objectives list; `src/misc/` has a from-scratch GP notebook.
