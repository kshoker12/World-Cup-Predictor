---
name: world-cup-predictor-spec
description: >-
  Implements the World Cup Predictor modular soccer prediction framework per
  the locked technical specification. Use when building, extending, or
  reviewing this project — data pipeline, Elo, features, LightGBM/CatBoost,
  calibration, ensemble, Monte Carlo simulation, or optional NN/Bayesian models.
---

# World Cup Predictor — Implementation Spec

Read [reference.md](reference.md) for the full locked specification. This file captures non-negotiable constraints and the build order.

## Locked Design Constraints

- **Gradient Boosting (LightGBM/CatBoost)** is the primary production model. Two separate Poisson models for λ_home and λ_away.
- **Neural Network** is experiment-only. International-first training; club pretrain is optional step 2.
- **Bayesian Dixon–Coles** is optional. When trained, contributes **λ (ensemble member) and ρ** via stored posterior means; replaces standalone ρ MLE.
- **Common interface**: all models output raw λ_home, λ_away before calibration.
- **No data leakage**: pre-match snapshots only; Elo updates after feature row is built.
- **Neutral venues**: Elo HA = 0; `is_home = 0` for features.
- Prioritize buildability, correctness, and simplicity.

## Locked Feature Defaults

| Feature | Definition |
|---------|------------|
| `elo_diff` | `Elo_home − Elo_away` (simple difference) |
| `form_diff` | Exponentially weighted points diff; decay=0.9, window=10, points 3/1/0 |
| `h2h_gd_weighted` | H2H GD with decay=0.85 per match (i=0 most recent) |
| `is_home` | 1 only if designated home AND not neutral |
| `tournament_importance` | WC=3, Qualifier=2, Friendly/other=1 |

## Locked Splits

| Split | Range |
|-------|-------|
| Train | `date < 2010-01-01` |
| Validation | `2010-01-01` – `2017-12-31` |
| Test | `date ≥ 2018-01-01` |
| WC backtest blocks | 2018 and 2022 World Cup finals (atomic) |

## Recommended Build Order

```
Implementation Progress:
- [x] Phase 1: Data pipeline + Elo + feature engineering
- [x] Phase 2: Gradient Boosting baseline + Monte Carlo simulator
- [x] Phase 3: Calibration + ensemble
- [x] Phase 4: Neural Network (optional experiment)
- [x] Phase 5: Bayesian model (optional, three-way ensemble)
```

### Phase 1 — Data & Features

**Verify before moving on:**
- Single central pipeline class; matches sorted by `(date, home_team, away_team)`
- Team name normalization table; dedup and invalid row drops
- Elo: initial 1500, K=20, HA +50 (non-neutral only)
- All 8 GBM features with locked formulas above
- Tests prove no leakage (feature at t uses only matches with date < t)

### Phase 2 — GBM + Simulator

**Verify before moving on:**
- Two LightGBM Poisson models (λ_home, λ_away)
- Match sampler: Poisson goals, DC adjustment, truncate 0–10
- Tournament engine: 32-team, 8×4 groups, standard FIFA knockout bracket
- Knockout ties: ET `λ * 1/3`, then penalties `P = 1/(1 + 10^(-elo_diff/400))`
- Backtest mode for 2018/2022; forecast mode for user-defined groups

### Phase 3 — Calibration & Ensemble

**Verify before moving on:**
- Multiplicative λ scaling (s_home, s_away) per model on validation
- Ensemble weights on simplex (GBM + optional NN + optional Bayesian)
- Standalone ρ MLE on validation low-score cells (skip if Bayesian implemented)
- W/D/L from final score grid

### Phase 4 — NN (Optional)

- International sequences first (length 10); club pretrain only if needed
- Must beat GBM validation deviance or document null result

### Phase 5 — Bayesian (Optional)

- Team attack/defense random effects + DC; data from 2000+
- Third ensemble member for λ; ρ from same fit
- Set `bayesian.use_in_ensemble: false` for ρ-only mode

## When Unclear

Stop and ask. Do not invent behavior not in [reference.md](reference.md).
