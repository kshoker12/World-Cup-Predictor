#!/usr/bin/env python3
"""Generate self-contained Kaggle notebooks with embedded project code."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

INCLUDE_PATHS = [
    "pyproject.toml",
    "src",
    "config",
    "scripts/run_kaggle_pipeline.py",
    "scripts/run_calibration_backtest_2018.py",
    "scripts/run_calibration_backtest_2022.py",
]

SHARED_PREFIX = '''\
import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ZIP_B64 = """{repo_b64}"""

PIP_PACKAGES = [
    "pandas>=2.0",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
    "lightgbm>=4.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "tqdm>=4.66",
    "torch>=2.0",
    "pymc>=5.0",
    "arviz>=0.16",
    "matplotlib>=3.7",
]

WORK = Path("/kaggle/working")
REPO = WORK / "WorldCupPredictor"
MODELS = WORK / "models"
DATA_ROOT: Path | None = None

REQUIRED_DATA_FILES = (
    "results.csv",
    "wc2026_results.csv",
    "former_names.csv",
    "fixtures.csv",
    "match_stats.csv",
    "understat_matches.parquet",
)


def install_dependencies() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *PIP_PACKAGES]
    )


def extract_project() -> Path:
    if REPO.exists():
        return REPO
    WORK.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode(REPO_ZIP_B64)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(WORK)
    if not REPO.exists():
        raise RuntimeError(f"Expected project at {{REPO}} after extract")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)]
    )
    return REPO


def discover_data_root() -> Path:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        raise FileNotFoundError(
            "/kaggle/input not found. Attach the soccer-data dataset to this notebook."
        )

    def score(root: Path) -> int:
        return sum(1 for name in REQUIRED_DATA_FILES if (root / name).exists())

    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for results_csv in kaggle_input.rglob("results.csv"):
        add(results_csv.parent)
    for item in sorted(kaggle_input.iterdir()):
        if item.is_dir():
            add(item)
            for sub in sorted(item.rglob("*")):
                if sub.is_dir():
                    add(sub)

    ranked = sorted(((score(path), path) for path in candidates), key=lambda x: (-x[0], str(x[1])))
    preferred = ("soccer-data", "soccer-data-dataset")
    for found_score, path in ranked:
        if found_score >= 3 and (
            path.name in preferred or any(part in preferred for part in path.parts)
        ):
            return path
    for found_score, path in ranked:
        if found_score >= 3:
            return path

    lines = ["Could not find soccer data files."]
    if not candidates:
        lines.append("/kaggle/input is empty. Click Add Data and attach soccer-data.")
    else:
        lines.append("/kaggle/input contents:")
        for path in candidates:
            files = sorted(p.name for p in path.iterdir() if p.is_file())
            lines.append(f"  {{path}}: {{', '.join(files[:8])}}")
    raise FileNotFoundError("\\n".join(lines))


def verify_data() -> Path:
    global DATA_ROOT
    DATA_ROOT = discover_data_root()
    missing = [name for name in REQUIRED_DATA_FILES if not (DATA_ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {{DATA_ROOT}}: {{missing}}")
    print("Data OK:", DATA_ROOT)
    return DATA_ROOT


def verify_models() -> Path:
    cal = MODELS / "calibration.json"
    if not cal.exists():
        raise FileNotFoundError(
            f"Missing {{cal}}. Run the WC 2026 training notebook first, "
            "or copy trained artifacts into /kaggle/working/models/."
        )
    print("Models OK:", MODELS)
    return MODELS
'''

WC2026_SUFFIX = '''\

FORECAST = MODELS / "wc2026_forecast.json"
REPORT = MODELS / "pipeline_report.json"


def run_pipeline(profile: str = "fast", resume: bool = False, seed: int = 42) -> int:
    repo = extract_project()
    data_root = verify_data()
    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_kaggle_pipeline.py"),
        "--profile", profile,
        "--seed", str(seed),
        "--data-root", str(data_root),
    ]
    if resume:
        cmd.append("--resume")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {{result.returncode}}")
    return result.returncode


def show_results(top_n: int = 5) -> None:
    if not FORECAST.exists():
        print(f"No forecast yet at {{FORECAST}}")
        return
    forecast = json.loads(FORECAST.read_text(encoding="utf-8"))
    print(f"Simulations: {{forecast.get('n_sims')}}")
    print("\\nTop champion probabilities:")
    for team, prob in sorted(
        forecast["champion_probs"].items(), key=lambda x: -x[1]
    )[:top_n]:
        print(f"  {{team}}: {{prob:.1%}}")
    print("\\nMost likely bracket champion:", forecast["most_likely_bracket"]["champion"])
    print("Path frequency:", f"{{forecast.get('most_likely_bracket_fraction', 0):.3%}}")
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        print(f"\\nElapsed hours: {{report.get('elapsed_hours')}} | Profile: {{report.get('profile')}}")


install_dependencies()
extract_project()
verify_data()
print("Setup complete. Next: run_pipeline('fast')")
'''

CALIBRATION_SUFFIX_TEMPLATE = '''\

YEAR = {year}
ACTUAL_CHAMPION = "{actual_champion}"
BACKTEST_JSON = MODELS / f"wc{{YEAR}}_calibration_backtest.json"
SCRIPT = REPO / "scripts" / f"run_calibration_backtest_{{YEAR}}.py"


def run_backtest(
    profile: str = "backtest",
    n_sims: int | None = None,
    seed: int = 42,
) -> int:
    """Robust full-tournament MC using trained models in /kaggle/working/models."""
    repo = extract_project()
    data_root = verify_data()
    verify_models()
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--profile", profile,
        "--seed", str(seed),
        "--models-dir", str(MODELS),
        "--data-root", str(data_root),
    ]
    if n_sims is not None:
        cmd.extend(["--sims", str(n_sims)])
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Backtest failed with exit code {{result.returncode}}")
    return result.returncode


def show_backtest_results(top_n: int = 5) -> None:
    if not BACKTEST_JSON.exists():
        print(f"No report yet at {{BACKTEST_JSON}}")
        return
    report = json.loads(BACKTEST_JSON.read_text(encoding="utf-8"))
    metrics = report["calibration_metrics"]
    print(f"WC {{YEAR}} calibration backtest — {{report['n_sims']}} simulations")
    print(f"Actual champion: {{metrics['actual_champion']}} ({{ACTUAL_CHAMPION}})")
    print(f"P(actual champion): {{metrics['p_actual_champion']:.4f}}")
    print(f"Champion rank: {{metrics['actual_champion_rank']}}")
    print(f"Multiclass Brier: {{metrics['multiclass_brier']:.4f}}")
    print(f"Log score: {{metrics['log_score']:.4f}}")
    print(f"Elapsed: {{report['elapsed_seconds']}}s")
    print("\\nTop champion probabilities:")
    for row in report["top_champion_probabilities"][:top_n]:
        print(f"  {{row['team']}}: {{row['prob']:.4f}}")
    print(f"\\nReport: {{BACKTEST_JSON}}")


install_dependencies()
extract_project()
verify_data()
try:
    verify_models()
except FileNotFoundError as exc:
    print("WARNING:", exc)
    print("Run the WC 2026 training notebook first, then re-run this cell.")
else:
    print("Setup complete. Next: run_backtest(profile='fast', n_sims=500)")
'''


def _build_repo_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE_PATHS:
            path = PROJECT_ROOT / rel
            if path.is_file():
                zf.write(path, f"WorldCupPredictor/{rel}")
            else:
                for file_path in path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in str(file_path):
                        arc = f"WorldCupPredictor/{file_path.relative_to(PROJECT_ROOT)}"
                        zf.write(file_path, arc)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    }


def _write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": _notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
        "outputs": [],
        "execution_count": None,
    }


def _md_cell(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> int:
    repo_b64 = _build_repo_zip_b64()

    wc_setup = SHARED_PREFIX.format(repo_b64=repo_b64) + WC2026_SUFFIX
    _write_notebook(
        NOTEBOOKS_DIR / "kaggle_wc2026.ipynb",
        [
            _md_cell([
                "# WC 2026 Kaggle Pipeline\n",
                "\n",
                "Attach **`soccer-data`** dataset. GPU + Internet ON.\n",
                "\n",
                "1. **Cell 1** — setup\n",
                "2. **Cell 2** — `run_pipeline('fast')` smoke test\n",
                "3. **Cell 3** — `run_pipeline('kaggle')` overnight training + forecast\n",
            ]),
            _code_cell(wc_setup),
            _code_cell(
                "# Smoke test (~3-15 min). Run before overnight job.\n"
                "run_pipeline(profile='fast')\n"
                "show_results()\n"
            ),
            _code_cell(
                "# Overnight full run (~8 hours).\n"
                "run_pipeline(profile='kaggle')\n"
                "show_results(top_n=10)\n"
            ),
            _code_cell(
                "# Resume after crash:\n"
                "# run_pipeline(profile='kaggle', resume=True)\n"
                "# show_results(top_n=10)\n"
            ),
        ],
    )

    for year, champion in ((2018, "France"), (2022, "Argentina")):
        cal_suffix = CALIBRATION_SUFFIX_TEMPLATE.replace("{year}", str(year)).replace(
            "{actual_champion}", champion
        )
        setup = SHARED_PREFIX.format(repo_b64=repo_b64) + cal_suffix
        _write_notebook(
            NOTEBOOKS_DIR / f"kaggle_calibration_backtest_{year}.ipynb",
            [
                _md_cell([
                    f"# WC {year} Calibration Backtest\n",
                    "\n",
                    "Robust full-tournament Monte Carlo to evaluate **trained model calibration**.\n",
                    "\n",
                    "**Prerequisite:** `/kaggle/working/models/` from the WC 2026 training notebook "
                    "(or attach a dataset with `calibration.json`, GBM, NN, Bayesian artifacts).\n",
                    "\n",
                    "Attach the same **`soccer-data`** dataset.\n",
                    "\n",
                    f"Actual champion: **{champion}**\n",
                    "\n",
                    "1. **Cell 1** — setup\n",
                    "2. **Cell 2** — quick smoke (`500` sims)\n",
                    "3. **Cell 3** — robust run (`50k` sims, ~1-3h)\n",
                ]),
                _code_cell(setup),
                _code_cell(
                    f"# Quick smoke (~5-20 min). Requires models from WC 2026 notebook.\n"
                    f"run_backtest(profile='fast', n_sims=500)\n"
                    f"show_backtest_results()\n"
                ),
                _code_cell(
                    f"# Robust calibration backtest (50k sims by default).\n"
                    f"run_backtest(profile='backtest')\n"
                    f"show_backtest_results(top_n=10)\n"
                ),
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
