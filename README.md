# World Cup Predictor

Modular soccer prediction framework for World Cup match forecasting.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional: neural network stack (Phase 4)
pip install -e ".[dev,nn]"
# Optional: Bayesian Dixon-Coles (Phase 5)
pip install -e ".[dev,bayesian]"
```

Place the martj42 international results CSV at `data/raw/results.csv`. Optional: `data/raw/former_names.csv` for team name aliasing.

---

## Phase 1: Data Pipeline

### Build features

```bash
python scripts/build_features.py
python scripts/validate_features.py
```

Writes `data/processed/features.parquet` with 8 GBM features, targets, and train/val/test split labels.

### Feature columns

| Column | Description |
|--------|-------------|
| `elo_diff` | Pre-match Elo home minus away |
| `gf_last_5_diff` | Rolling avg goals scored diff (last 5) |
| `ga_last_5_diff` | Rolling avg goals conceded diff (last 5) |
| `form_diff` | Exponentially weighted form points diff |
| `is_home` | 1 if home and not neutral |
| `is_neutral` | 1 if neutral venue |
| `tournament_importance` | 1 friendly / 2 qualifier / 3 World Cup |
| `h2h_gd_weighted` | Weighted head-to-head goal difference |
| `home_score`, `away_score` | Targets |
| `split` | `train` / `val` / `test` |

### Splits

- **train**: date < 2010-01-01
- **val**: 2010-01-01 to 2017-12-31
- **test**: date >= 2018-01-01

---

## Phase 2: GBM + Monte Carlo Simulator

### Train models

```bash
python scripts/train_gbm.py
python scripts/validate_model.py
```

Trains two LightGBM Poisson models (`λ_home`, `λ_away`) on the train split. Saves to `data/models/`.

### Run backtest

```bash
python scripts/run_backtest.py --year 2018 --sims 500
python scripts/run_backtest.py --year 2022 --sims 10000
```

### Calibration backtests (robust MC on trained models)

After training (`run_kaggle_pipeline` or local fit), evaluate calibration on held-out World Cups:

```bash
# Default: 50k sims (config/profiles/backtest.yaml)
python scripts/run_calibration_backtest_2018.py
python scripts/run_calibration_backtest_2022.py

# Quick smoke
python scripts/run_calibration_backtest_2018.py --profile fast --sims 200

# On Kaggle after overnight run (models in /kaggle/working/models)
python scripts/run_calibration_backtest_2022.py --models-dir output/
```

Writes `wc2018_calibration_backtest.json` / `wc2022_calibration_backtest.json` with champion probs, advancement probs, `P(actual champion)`, multiclass Brier, and log score.

By default uses **knockout-only** simulation (actual R16 bracket, same `KnockoutSimulator` engine as the WC 2026 forecast — ~5h for 50k sims on GPU). Pass `--full-tournament` for the slower group-stage + knockout path (~27h for 50k).

### Run forecast

```bash
python scripts/run_forecast.py --config config/tournaments/world_cup_2022.yaml --sims 10000
```

### Validate simulation

```bash
python scripts/validate_simulation.py --year 2018 --sims 500
python scripts/validate_simulation.py --year 2022 --sims 500
```

Checks: 48 group + 15 knockout matches per sim, champion probs sum to 1, actual champion has P > 0, cumulative advancement bounds, reproducibility.

---

## Phase 3: Calibration + Dixon–Coles

Fits multiplicative λ scaling and Dixon–Coles ρ on the validation split, then wires the calibrated predictor into simulation and evaluation.

### Fit calibration

```bash
python scripts/fit_calibration.py
python scripts/validate_calibration.py
```

Writes `data/models/calibration.json` (`scaling_gbm`, `scaling_nn`, `scaling_bayesian`, `ensemble`, `dixon_coles`). Backtest and forecast scripts load this automatically.

**Validation gates** (`validate_calibration.py`):

- Artifact exists with positive scaling factors
- Calibrated val Poisson deviance ≤ raw GBM on validation
- `-0.2 ≤ rho ≤ 0.1`
- W/D/L probabilities sum to 1 on validation rows
- Test metrics are reported only (not used for tuning)

### Evaluate predictions

```bash
python scripts/evaluate_predictions.py
```

Prints val/test comparison table: Poisson deviance, W/D/L log loss, Brier, ECE for raw GBM vs calibrated.

Typical improvement on validation: modest drop in goal deviance and W/D/L calibration error (ECE).

### Backtest with calibration

```bash
python scripts/run_backtest.py --year 2018 --sims 500
```

Uses `CalibratedPredictor` and fitted ρ in the score grid. Tournament champion odds may shift; match-level probabilistic quality is the primary gate.

### Test

```bash
pytest -v -p no:rerunfailures -p no:seleniumbase
```

65 tests covering Phases 1–5 (club loader, sequences, NN model, calibration ensemble, Bayesian).

---

## Phase 4: Neural Network — Club Pretrain + International Fine-tune

Two-stage LSTM: pretrain on merged club matches (Understat + soccer-dataset), fine-tune on international sequences, then ensemble with calibrated GBM.

### Install (includes PyTorch + soccerdata)

```bash
pip install -e ".[dev,nn]"
```

On Apple Silicon, training uses **MPS** when available (`nn.device: mps` in `config/defaults.yaml`).

### Club data setup (both sources required)

**Source A — Understat match-level** (network once, cached):

```bash
python scripts/fetch_understat_matches.py
```

If the soccerdata TLS fetch fails in your environment, use the Big-5 proxy from soccer-dataset:

```bash
python scripts/fetch_understat_matches.py --fallback-from-club
```

**Source B — soccer-dataset** ([eatpizzanot/soccer-dataset](https://github.com/eatpizzanot/soccer-dataset)):

Download `csv/fixtures.csv`, `csv/match_stats.csv`, `csv/teams.csv`, and `csv/leagues.csv` into `data/raw/club/`.

```bash
mkdir -p data/raw/club
# place fixtures.csv, match_stats.csv, teams.csv, leagues.csv
python scripts/validate_club_data.py   # fails if either source missing
python scripts/build_club_matches.py   # merge + dedupe → club_matches.parquet (≥10k rows)
python scripts/build_club_sequences.py
```

> **Note:** Kaggle player-season Understat files (`2014-2015.csv` …) are the wrong granularity for LSTM pretrain — use match-level sources above.

### International sequences + training

```bash
python scripts/build_nn_sequences.py
python scripts/train_nn.py --phase all          # pretrain → nn_pretrain.pt, finetune → nn_model.pt
python scripts/validate_nn.py                   # GBM vs NN vs ensemble on intl val
python scripts/fit_calibration.py               # scaling_gbm, scaling_nn, ensemble, rho
python scripts/evaluate_predictions.py
python scripts/run_backtest.py --year 2018 --sims 500 --no-progress
```

### Success criterion

`validate_nn.py` reports **PASS** if calibrated ensemble val Poisson deviance ≤ GBM-only. A **null result** (NN does not beat GBM) is valid — club→national transfer may not help.

### Progress bars

Long scripts use `tqdm` via `worldcup_predictor.utils.progress`. Pass `--no-progress` to disable (CI/backtests).

---

## Phase 5: Bayesian Dixon–Coles (optional, three-way ensemble)

Hierarchical PyMC model with team attack/defense random effects and the same 8 GBM features. When trained, Bayesian contributes **λ and ρ** to production via a third ensemble member alongside GBM and NN.

### Install

```bash
pip install -e ".[dev,bayesian]"
```

Training runs on **CPU** (PyMC NUTS), not MPS. Inference uses stored posterior means only (no PyMC at predict time).

### Train and validate

```bash
python scripts/train_bayesian.py
python scripts/validate_bayesian.py
python scripts/fit_calibration.py
python scripts/validate_calibration.py
python scripts/evaluate_predictions.py
```

**Important:** After upgrading to three-way ensemble, re-run `train_bayesian.py` so `bayesian.json` includes `att_mean` / `def_mean` (required for λ inference).

Fits on international matches from **2000-01-01** through the validation window. Default `split_scope: train` (train split only) for fair ensemble comparison; set `split_scope: train_val` for legacy train+val fits.

### Role in production

| Component | Source |
|-----------|--------|
| λ_home, λ_away | Weighted GBM + NN + Bayesian (validation-tuned simplex) |
| Dixon–Coles ρ | Bayesian posterior mean when `bayesian.json` exists, else MLE |
| Team attack/defense | Used at inference via `BayesianPredictor` |

Set `bayesian.use_in_ensemble: false` in `config/defaults.yaml` to use Bayesian ρ only (legacy ρ-only mode).

`calibration.json` stores `scaling_gbm`, `scaling_nn`, `scaling_bayesian`, `ensemble` (`w_gbm`, `w_nn`, `w_bayesian`), `min_ensemble_weight`, and `dixon_coles`.

---

## Kaggle pipeline (unified run)

Single orchestrator for local smoke tests and full Kaggle GPU runs:

```bash
pip install -e ".[dev,bayesian,nn]"

# Local smoke (~5–15 min): small epochs, 200 sims
python scripts/run_kaggle_pipeline.py --profile fast

# Kaggle full run (8h budget, club pretrain, GPU NN, budget-scaled sims)
python scripts/run_kaggle_pipeline.py --profile kaggle

# Resume after partial run (skips existing artifacts)
python scripts/run_kaggle_pipeline.py --profile kaggle --resume
```

### What it does

1. Merges `data/raw/results.csv` + `data/raw/wc2026_results.csv` (WC 2026 through R32)
2. Builds club sequences when available; **club pretrain → international finetune** on GPU
3. **gate_gbm**: trains GBM plain vs GBM+NN embeddings; picks lower val deviance
4. Trains Bayesian on **train split only** (`split_scope: train`), scaled to leave ~2.7h for forecast
5. Fits calibration with **10% minimum ensemble weight** per active model
6. Runs **R16→Final** Monte Carlo; sim count fills remaining budget (cap 200k on kaggle profile)

### Outputs (`data/models/` locally, `/kaggle/working/models/` on Kaggle)

| File | Description |
|------|-------------|
| `gbm_home.txt`, `gbm_away.txt` | Production GBM (ablation winner) |
| `nn_model.pt`, `nn_pretrain.pt`, `nn_meta.json` | Trained NN |
| `bayesian.json` | Bayesian posterior summary |
| `calibration.json` | Scaling, min-weight ensemble, ρ |
| `pipeline_report.json` | Timings, gate_gbm, weights, planned `n_sims` |
| `wc2026_forecast.json` | Champion/advancement probs, **per-match win probs**, **most likely bracket**, one random sample bracket |

`wc2026_forecast.json` fields:
- **`champion_probs`** / **`advancement_probs`**: aggregate tournament outcomes
- **`match_win_probs`**: for each simulated fixture, `p_home_win` / `p_away_win` (later-round pairings weighted by how often they occur)
- **`most_likely_bracket`**: mode full knockout path across all sims (not the same as `sample_bracket`, which is sim #0 only)
- **`most_likely_bracket_fraction`**: how often that exact path occurred

### Kaggle setup (self-contained notebook)

**Data dataset** (`soccer-data`) — flat layout at `/kaggle/input/soccer-data/`:

| File | Purpose |
|------|---------|
| `results.csv` | International matches |
| `wc2026_results.csv` | WC 2026 results through R32 |
| `former_names.csv` | Team name aliases |
| `fixtures.csv` | Club fixtures (NN pretrain) |
| `match_stats.csv` | Club match stats |
| `understat_matches.parquet` | Understat club matches |

**Notebook** — upload from `notebooks/` (regenerate after code changes):

```bash
python scripts/generate_kaggle_notebook.py
```

| Notebook | Purpose |
|----------|---------|
| `kaggle_wc2026.ipynb` | Train models + WC 2026 knockout forecast |
| `kaggle_wc2026_forecast.ipynb` | **Forecast only** — load models from `soccer-train`, run 200k MC (no training) |
| `kaggle_calibration_backtest_2018.ipynb` | Knockout-only MC backtest vs France 2018 (same engine as forecast) |
| `kaggle_calibration_backtest_2022.ipynb` | Knockout-only MC backtest vs Argentina 2022 (same engine as forecast) |

### Forecast only (models already trained)

Use **`kaggle_wc2026_forecast.ipynb`** when training is done and you only need `wc2026_forecast.json`.

1. Upload your trained artifacts as Kaggle dataset **`soccer-train`** (files from `output/`: `calibration.json`, `gbm_*.txt`, `nn_model.pt`, `bayesian.json`, etc.)
2. Attach **`soccer-train`** + **`soccer-data`**, GPU ON
3. Run Cell 1 (setup) → Cell 2 (500 sim smoke) → Cell 3 (**200,000 sims**, fixed)

Do **not** re-run `kaggle_wc2026.ipynb` for forecast — it retrains from scratch (~8h).

Outputs: `/kaggle/working/models/wc2026_forecast.json` and `forecast_report.json`.

Local forecast smoke:

```bash
python scripts/run_kaggle_forecast.py \
  --models-dir output/ \
  --data-root tests/fixtures/flat_soccer_data \
  --sims 10
```

### GitHub Pages (live forecast bracket)

**Live URL:** https://kshoker12.github.io/World-Cup-Predictor/

Static files live in [`docs/`](docs/) (`index.html` + `data/wc2026_forecast.json`).

**One-time setup** (repo owner): GitHub → **Settings** → **Pages** → **Build and deployment** → Source: **Deploy from a branch** → Branch **`main`** / folder **`/docs`** → Save. The site is live within ~1 minute.

After a new forecast run, refresh the site:

```bash
cp output/wc2026_forecast.json docs/data/wc2026_forecast.json
python scripts/show_forecast.py docs/data/wc2026_forecast.json --html docs/index.html --no-print
git add docs/ && git commit -m "Update forecast page" && git push
```

Optional: switch Pages source to **GitHub Actions** and run [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml) manually to regenerate HTML in CI before deploy.

WC 2026 training notebook cells:

1. **Setup** — installs pip deps, extracts embedded project code, defines `run_pipeline()` / `show_results()`
2. **Smoke test** — `run_pipeline(profile="fast")` (~3–15 min)
3. **Overnight** — `run_pipeline(profile="kaggle")` (~8 h)

Calibration notebooks (attach **`soccer-train`** or **`output`** + **`soccer-data`**; models staged from your local `output/` upload):

1. **Setup** — stage models from attached dataset + `run_backtest()` / `show_backtest_results()`
2. **Smoke** — `run_backtest(profile="fast", n_sims=500)`
3. **Robust** — `run_backtest(profile="backtest")` (50k sims)

Settings: **GPU** accelerator, **Internet ON**.

Outputs: `/kaggle/working/models/wc2026_forecast.json` and `pipeline_report.json`.

Local flat-layout test:

```bash
python scripts/build_flat_test_data.py
python scripts/run_kaggle_pipeline.py --profile fast --data-root tests/fixtures/flat_soccer_data
```

Profiles live in `config/profiles/fast.yaml` and `config/profiles/kaggle.yaml`.

---

## Tournament configs

Historical World Cups: `config/tournaments/world_cup_2018.yaml`, `world_cup_2022.yaml`.

WC 2026 knockout (R16 onward): `config/tournaments/world_cup_2026_knockout.yaml`.

Knockout rules: extra time at `λ × 1/3`, penalties via Elo-weighted logistic. Production λ and ρ come from `calibration.json` (three-way ensemble when Bayesian artifact exists).
