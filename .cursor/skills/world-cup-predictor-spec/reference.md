**Final Locked Technical Specification**

### Project Overview & Objectives

**Project Name**: World Cup Predictor (Modular Soccer Prediction Framework)

**Primary Goal**: Build a practical, extensible system that predicts FIFA World Cup matches by modeling goal distributions and simulates full tournament progressions using well-calibrated probabilistic outputs.

**Core Objectives**:
- Build a modular soccer prediction framework focused on realistic goal distributions and tournament simulations.
- Use the Neural Network as an **optional experiment-only** feature extractor from richer match data.
- Combine engineered features and probabilistic modeling to produce usable predictions and simulations.
- Support accurate and internally consistent tournament simulation by sampling scorelines from ensembled goal distributions.
- Create reusable components that can be extended to other competitions.

### Design Principles

- The **Gradient Boosting** model is the **primary and most reliable production component**.
- The **Neural Network** acts as an **optional experiment-only** feature extractor from richer match data. It is not part of the production pipeline.
- The **Bayesian** model is an **optional** refinement for team-strength hierarchy and low-score correlation. When trained, it contributes λ to the three-way ensemble and replaces standalone ρ MLE.
- **Expected goal rates (λ_home, λ_away)** serve as the common prediction interface. All models output raw expected goal intensities before calibration.
- Goal distributions are modeled using Poisson-based processes with optional correlation adjustments (Dixon–Coles).
- All models use only information available **before kickoff**.
- Strict chronological processing must be manually enforced in code to prevent data leakage.
- The system prioritizes **buildability, correctness, and simplicity**.
- The architecture is designed for **extensibility**, with awareness that additional model complexity yields diminishing returns.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA LAYER                               │
│  • Kaggle International Matches (martj42)                       │
│  • Club Match Datasets (team-level stats) — Neural Network only │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              DATA & RATINGS LAYER                               │
│  • Data Cleaning & Validation                                   │
│  • Elo Rating Computation (reusable module)                     │
│  • Rolling Form & Momentum Computation                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING LAYER                            │
│  • Model-specific feature views                                 │
│  • Rolling Pre-Match Performance Vectors (Neural Network only)  │
│  • Rating + Context + Strength Features (GBM + Bayesian)        │
│  Output: Model-specific features + Optional NN Embeddings       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PREDICTION LAYER                                │
│                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────┐          │
│  │ Neural Network       │     │ Gradient Boosting    │          │
│  │ (Experiment-only)    │     │ (Primary Model)      │          │
│  │ Encoder → Embedding  │────▶│ Predicts λ_home,     │          │
│  │ → Expected Goal Rate │     │ λ_away + Optional    │          │
│  │   (λ) Prediction Head│     │ Embeddings           │          │
│  └──────────────────────┘     └──────────────────────┘          │
│                                                                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Bayesian Hierarchical Dixon–Coles (Optional)       │         │
│  │ (Engineered features only – no embeddings)         │         │
│  └────────────────────────────────────────────────────┘         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│         CALIBRATION & ENSEMBLE LAYER                            │
│  • Per-model multiplicative λ calibration (validation fit)      │
│  • Weighted ensemble of GBM (+ optional NN) λ values            │
│  • Dixon–Coles ρ (standalone or via Bayesian model)           │
│  • Final predictive goal distribution                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│           MONTE CARLO TOURNAMENT SIMULATION LAYER               │
│  • Sample scorelines from ensembled goal distribution           │
│  • Derive win/draw/loss, goal difference, tiebreakers           │
│  • Outputs: Champion %, advancement probabilities, bracket stats│
└─────────────────────────────────────────────────────────────────┘
```

### Data Layer

#### Primary Dataset (Used by All Models)

**Name**: International Football Results (1872–2026)  
**Kaggle Link**: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

**Raw Columns Used**:
- `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral`

**Core Modeling Unit**:
One international match = one training example.  
Target = `(home_goals, away_goals)` used to supervise expected goal rates λ.

**Target Definition**:
Both λ_home and λ_away are trained jointly from the same match sample using independent Poisson Negative Log-Likelihood heads (sum of two Poisson NLLs).

#### Data Cleaning & Validation Rules

Apply before feature computation:

- Parse `date` to UTC date; drop rows with unparseable dates.
- Drop rows where `home_team == away_team`, scores are null, or scores are negative.
- Cast scores to non-negative integers.
- Normalize team names via a canonical mapping table (e.g. `Czech Republic` → `Czechia`, `IR Iran` → `Iran`). Maintain mappings in version-controlled config; update as new aliases appear.
- Deduplicate on `(date, home_team, away_team, home_score, away_score)` keeping first occurrence.
- Sort all matches globally by `(date, home_team, away_team)` before chronological processing.

#### Club Dataset (Neural Network Only — Optional Experiment)

**Purpose**: Provide richer team-level performance data for the Neural Network (optional experiment).

**Dataset**: https://www.kaggle.com/datasets/jashsheth5/indepth-soccer-statistics-xg-xa-and-more

**How Club Data Will Be Used**:
- Aggregate into team-level rolling performance vectors (fixed windows of 5 and 10 matches).
- Used **only** for the Neural Network.
- Features are a subset of available advanced stats depending on league coverage.
- Missing values are forward-filled within league boundaries. Leagues are normalized independently.

**Club → National Transfer Note**:
Club data covers club teams, not national teams. Pre-training learns general football dynamics, not direct national-team form. v1 NN experiment should **first** train on international sequences only; club pre-training is a second-step experiment if international-only results are insufficient.

### Feature Engineering Layer

#### Features for Gradient Boosting & Bayesian Model (Flat Vector)

| Feature | Definition | Default / Encoding |
|---------|------------|-------------------|
| `elo_diff` | `Elo_home − Elo_away` at pre-match snapshot | Simple difference (not margin-weighted) |
| `gf_last_5_diff` | `avg_goals_scored_home(last 5) − avg_goals_scored_away(last 5)` | Last 5 matches by count per team |
| `ga_last_5_diff` | `avg_goals_conceded_home(last 5) − avg_goals_conceded_away(last 5)` | Last 5 matches by count per team |
| `form_diff` | `form_points_home − form_points_away` | See formula below |
| `is_home` | Effective home flag | `1` if designated home **and** not neutral; else `0` |
| `is_neutral` | From dataset | `1` if neutral venue; else `0` |
| `tournament_importance` | Ordinal match stakes | World Cup = 3, Qualifier = 2, Friendly = 1, other = 1 |
| `h2h_gd_weighted` | Exponentially weighted H2H goal difference | See formula below |

**`form_diff` Formula** (locked default):

```
For each team, over its last 10 matches (most recent first, index i = 0..9):
  result_points = 3 (win), 1 (draw), 0 (loss)
  form_points = Σ (0.9^i) * result_points
form_diff = form_points_home − form_points_away
```

**`h2h_gd_weighted` Formula** (locked default):

```
For all prior H2H matches between the two teams (chronological, pre-match only):
  gd_i = goals_for_team − goals_against_team  (from that team's perspective)
  weight_i = 0.85^i   where i = 0 is most recent H2H match
h2h_gd_weighted = Σ (weight_i * gd_i)   (signed from home team perspective)
```

Configurable overrides: `form_decay=0.9`, `form_window=10`, `h2h_decay=0.85` in project config.

**`tournament_importance` Mapping Rules**:
- `tournament` contains `"World Cup"` (and not `"qualif"`) → 3
- `tournament` contains `"qualif"` (case-insensitive) → 2
- `tournament` contains `"Friendly"` → 1
- All others → 1

**Elo Rules**:
- Initial Elo = 1500
- K-factor = 20
- Home advantage = +50 Elo points added to home team **only when `neutral=False`**
- Standard logistic expectation (goal-independent update by default)
- **Elo update happens strictly after the match is consumed for feature computation** (feature row always uses Elo(t−1))

**Neutral Venue Handling** (locked):
- When `neutral=True`: Elo home advantage = 0; `is_home = 0` for feature purposes.
- The `home_team` column is retained for bookkeeping and H2H pairing only.

**Chronological Processing Rule**:
All rolling features and Elo values for a match at time *t* must be computed using **only matches with date < t**, processed strictly chronologically per team. **This rule must be manually enforced and tested in code.**

**Rolling Features Window Semantics**:
- “Last N matches” = last N matches by match count (not calendar days).
- Teams with fewer than N matches use available history only (no imputation from future).
- Early-season bias is accepted and handled via validation.

#### Features for Neural Network (Sequence View) — Optional Experiment

Each team is represented as a sequence of recent performance vectors containing:
- Goals scored / conceded
- Opponent Elo
- Days since last match
- Rolling xG/xGA, shots, deep completions, PPDA (where available)

Default sequence length: **10 matches**. Pad shorter histories with zeros; never pad from future.

**Embeddings**:
Neural Network embeddings are generated as fixed-length team representations using only past data. They are **optional experiment-only** for Gradient Boosting and must be produced only on training-period data to avoid leakage. They are not part of the core production pipeline.

### Prediction Layer

#### Gradient Boosting Model (LightGBM / CatBoost) — Primary Model

- **Objective**: Poisson objective.
- **Input**: Flat match-level feature vector (Elo diff, form diffs, context, optional NN embeddings).
- Predicts expected goal rates (λ_home, λ_away).
- This is the main production model.

**Training Pattern** (locked):
- Train **two separate** Poisson models on the same feature matrix: one for `λ_home`, one for `λ_away`.
- Joint loss = NLL_home + NLL_away (mathematically equivalent to two single-target models).
- Default library: LightGBM; CatBoost is an acceptable alternative.

#### Neural Network (PyTorch) — Optional Experiment

- **Training Strategy (v1)**: Train on international sequences first. Club pre-training is an optional second experiment.
- **Objective**: Predict expected goal rates (λ_home, λ_away) using Poisson Negative Log-Likelihood.
- **Architecture**: Encoder (LSTM or Transformer) → Latent Embedding → Prediction Head.
- Embeddings can be extracted and optionally passed to Gradient Boosting after validation (experiment only).

**Success criterion for NN experiment**: Must improve validation Poisson deviance vs GBM-only baseline, or results are documented as null.

#### Bayesian Hierarchical Dixon–Coles Goal Model — Optional Refinement

- Uses only low-dimensional engineered features plus team-level attack/defense random effects.
- Models joint goal distribution with Dixon–Coles correlation.
- Explicitly excludes Neural Network embeddings.
- Provides uncertainty quantification (optional to implement).
- Fit on matches from **2000-01-01 onward** by default (configurable) to keep runtime practical.
- **Role in v1**: If implemented, outputs λ_home/λ_away (ensemble member) and DC correlation ρ; **replaces** standalone ρ estimation. Included in three-way ensemble when `use_in_ensemble: true`.

**Modeling Assumption**:
All non-Bayesian models assume independent Poisson goal rates. Correlation is handled via Dixon–Coles in the calibration layer (or via the Bayesian model if implemented).

### Calibration & Ensemble Layer

#### Per-Model λ Calibration (v1 — locked)

Multiplicative scaling fit on the validation set:

```
λ_home_cal = s_home * λ_home_raw
λ_away_cal = s_away * λ_away_raw
```

- Fit `s_home`, `s_away ≥ 0` per model by minimizing validation Poisson deviance.
- Apply separately to GBM, (if used) NN, and (if used) Bayesian before ensembling.

#### Ensemble (v2 — implemented)

- Ensemble **GBM**, **optional NN**, and **optional Bayesian** calibrated λ values.
- Optimize weights `w = (w_gbm, w_nn, w_bayesian)` on the simplex (`w ≥ 0`, `Σw = 1`) minimizing validation Poisson deviance.
- If only GBM is available: `w_gbm = 1`.
- Disable Bayesian λ via config `bayesian.use_in_ensemble: false` (ρ-only fallback).

#### Dixon–Coles Correlation (v1 — locked)

When Bayesian model is **not** implemented:
- Fit ρ on validation-set low-score cells (0–0, 0–1, 1–0, 1–1) via MLE using ensembled λ values.
- Apply DC adjustment to the score grid before W/D/L and simulation.

When Bayesian model **is** implemented:
- Use ρ from the Bayesian posterior; skip standalone ρ fitting.
- λ from Bayesian enters the ensemble when `use_in_ensemble` is true.

#### Outputs

- Win/Draw/Loss probabilities derived from the final DC-adjusted (or independent Poisson) score grid.
- Score grid truncated at 0–10 goals per team for computational efficiency (approximation only).

### Monte Carlo Tournament Simulation Layer

#### Match-Level Simulation (locked)

- Sample full scorelines from the final ensembled goal distribution.
- **10,000 simulations per fixture** (configurable).
- Goals sampled from Poisson(λ) per team; truncate at 0–10 for efficiency.
- Apply Dixon–Coles adjustment to low-score joint probabilities when ρ is available.

**Knockout tie resolution** (locked):

| Stage | Rule |
|-------|------|
| Extra time | `λ_ET = λ * (1/3)` (30 min / 90 min); sample one ET period per team |
| Penalties (if still tied) | `P(home wins) = 1 / (1 + 10^(-elo_diff/400))` using pre-match Elo diff |

#### World Cup Tournament Format (v1 — locked)

Backtesting and simulation use the **32-team, 8-group format** (2018/2022 structure):

**Group stage**:
- 8 groups (A–H) of 4 teams each.
- Round-robin: 6 matches per group.
- Points: 3 win, 1 draw, 0 loss.
- Tiebreakers (in order): goal difference → goals scored.
- Top 2 per group advance (16 teams).

**Knockout bracket** (standard FIFA pairing):
- Round of 16: `A1–B2, C1–D2, E1–F2, G1–H2, B1–A2, D1–C2, F1–E2, H1–G2`
- Quarter-finals → Semi-finals → Final.
- All knockout matches use ET + penalties rule above on draws.

**Operating modes**:
1. **Backtest mode**: Load actual group assignments and bracket for a historical World Cup (2018, 2022); simulate all matches; compare to actual outcomes.
2. **Forecast mode**: Accept user-defined groups and team list; run full Monte Carlo tournament from group draw through final.

**Outputs**:
- Champion probability per team
- Advancement probabilities (group exit, each knockout round)
- Bracket statistics (most likely paths, upset rates)

### Evaluation Framework

**Primary Metrics**:
- Poisson deviance / goal log-likelihood
- RMSE & MAE on predicted vs observed goals

**Derived Metrics**:
- Log Loss, Brier Score, Expected Calibration Error on win/draw/loss

**Model Comparison**:
- Individual models vs Final Ensemble + Historical World Cup backtesting

**Train / Val / Test Split** (locked defaults):

| Split | Date Range | Notes |
|-------|------------|-------|
| Train | All matches with `date < 2010-01-01` | Model fitting |
| Validation | `2010-01-01` ≤ `date` < `2018-01-01` | Calibration, ensemble weights, ρ tuning, hyperparameter selection |
| Test | `date` ≥ `2018-01-01` | Final evaluation only; never used for tuning |

**Atomic tournament holdouts for World Cup backtesting**:
- **2018 World Cup**: all matches where `tournament` indicates 2018 World Cup finals
- **2022 World Cup**: all matches where `tournament` indicates 2022 World Cup finals

Entire tournaments are held out as atomic blocks — no partial tournament leakage. Qualifier matches in the test window may be used for match-level metrics but World Cup backtest reports are computed on finals-only blocks.

### Tech Stack & Development Guidelines

- **Data Pipeline**: Single central class with strict chronological processing (**manual verification required**).
- **Models**: PyTorch, LightGBM/CatBoost, PyMC or CmdStanPy (Bayesian optional).
- **Config**: Version-controlled YAML/JSON for Elo params, feature decay defaults, split dates, simulation counts.
- **Recommended Development Order** (for successful vibe coding):
  1. Data pipeline + feature engineering + Elo
  2. Gradient Boosting baseline + Monte Carlo Simulator
  3. Calibration + Ensemble
  4. Neural Network (optional experiment)
  5. Bayesian model (optional)

### Future Work

- Distribution-level ensembling.
- Further representation learning refinements.
- Extension to other competitions (e.g. 48-team World Cup format).
- Richer group-stage tiebreakers (head-to-head, fair play).
- Isotonic or beta calibration on W/D/L derived from score grid.

---

**This specification is final and locked** (amended post-feasibility audit).

All major implementation ambiguities have been resolved with locked defaults. Configurable parameters live in project config; do not change locked behavior without updating this document.
