# FeatherSim — LEARNINGS.md

Append a dated entry whenever something surprises you (sim quirks, training failures, flaky
tests). This is the compounding mechanism: never get bitten by the same issue twice.

## 2026-06-18 — Dev environment baseline
Python 3.13.5 (anaconda), pip 25.1, pytest 8.3.4, git 2.50.1 on macOS (darwin 24.6).

## 2026-06-18 — PyBullet on Python 3.13 — VERIFY in Phase 1
PyBullet wheels have historically lagged new CPython releases. Before building Phase 1, confirm
`pip install pybullet` succeeds on 3.13; if not, options are (a) a conda-forge build, (b) pin a
compatible Python via a venv. Don't assume the install is clean. [resolve in Phase 1]

## 2026-06-18 — pytest import path
Tests import the top-level `feathersim` package. Set `pythonpath = ["."]` in
`[tool.pytest.ini_options]` so `pytest` from repo root puts the root on `sys.path` (avoids
`ModuleNotFoundError` without needing an editable install or a root `conftest.py`).
