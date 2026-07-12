# World Cup Predictor

Round-by-round knockout forecasting for the **2026 FIFA World Cup**.

**[Live forecast](https://kshoker12.github.io/World-Cup-Predictor/)** · **[Write-up](https://kshoker12.github.io/portfolio/world-cup-predictor.html)**

## What it does

Three models each predict expected goals \((\lambda_h, \lambda_a)\) for a match:

| Model | Role | Ensemble weight |
|-------|------|----------------:|
| LightGBM | Tabular features (Elo, form, H2H, context) + LSTM embeddings | 0.43 |
| LSTM | Last 10 matches per team | 0.33 |
| Bayesian | Hierarchical attack/defense (Dixon–Coles) | 0.24 |

Calibration scales and combines those rates. A Dixon–Coles score grid turns \((\lambda_h, \lambda_a)\) into scoreline probabilities. Knockout ties go through extra time, then an Elo-based penalty draw.

At the start of each knockout round, new results are ingested and **80,000** bracket simulations are run from the current tree. Probabilities are event counts: \(\hat{P}(E) = \#\{sims where E\} / N\).

## Results so far

Held-out test (8,192 matches): ensemble beats every single model (Poisson deviance 2.35, outcome accuracy 60.4%).

Round of 16: **6/8** correct. Live QF title odds and match picks are on the [site](https://kshoker12.github.io/World-Cup-Predictor/).

## Layout

```
src/worldcup_predictor/   models, features, calibration, simulation
scripts/                  train, calibrate, forecast, evaluate
config/                   profiles (kaggle, fast, …) + tournament YAML
data/                     raw results → processed features/sequences
output/                   production artifacts (gbm, nn, bayesian, calibration)
docs/                     GitHub Pages forecast JSON
write-up.md               full technical write-up
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,nn,bayesian]"
```

Put international results at `data/raw/results.csv` ([martj42](https://github.com/martj42/international_results)). Optional: `data/raw/former_names.csv`, club data for LSTM pretrain.

## Common commands

```bash
# Full production train (Kaggle profile)
python scripts/run_kaggle_pipeline.py --profile kaggle

# WC 2026 knockout forecast (80k sims)
python scripts/run_kaggle_forecast.py          # from R16
python scripts/run_kaggle_forecast_qf.py       # from QF
python scripts/run_kaggle_forecast_sf.py       # from SF
# Or use notebooks/kaggle_wc2026_forecast{,_qf,_sf}.ipynb on Kaggle

# Held-out test metrics
python scripts/evaluate_test_metrics.py --models-dir output
```

Faster smoke runs use `--profile fast`. Tournament fixtures live in `config/tournaments/`.

## Data note

Features and sequences use only information available **before** kickoff. The same chronological pipeline is reused inside each Monte Carlo simulation so later rounds see updated Elo/form from earlier simulated results.
