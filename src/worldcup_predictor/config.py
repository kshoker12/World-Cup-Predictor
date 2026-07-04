from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class EloConfig:
    initial: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 50.0


@dataclass(frozen=True)
class FeatureConfig:
    form_decay: float = 0.9
    form_window: int = 10
    rolling_goals_window: int = 5
    h2h_decay: float = 0.85


@dataclass(frozen=True)
class SplitConfig:
    train_end: date
    val_end: date


@dataclass(frozen=True)
class GBMConfig:
    num_boost_round: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class SimulationConfig:
    n_sims: int = 10000
    max_goals: int = 10
    dixon_coles_rho: float = 0.0


@dataclass(frozen=True)
class CalibrationConfig:
    rho_min: float = -0.2
    rho_max: float = 0.1
    scaling_bounds: tuple[float, float] = (0.5, 2.0)
    min_ensemble_weight: float = 0.10


@dataclass(frozen=True)
class ClubConfig:
    data_dir: str = "data/raw/club"
    min_date: date = date(2014, 1, 1)
    leagues: tuple[str, ...] = ()
    forward_fill_within_league: bool = True
    understat_leagues: tuple[str, ...] = (
        "ENG-Premier League",
        "ESP-La Liga",
        "GER-Bundesliga",
        "ITA-Serie A",
        "FRA-Ligue 1",
    )
    understat_seasons: tuple[str, ...] = (
        "1415",
        "1516",
        "1617",
        "1718",
        "1819",
    )


@dataclass(frozen=True)
class BayesianConfig:
    min_date: date = date(2000, 1, 1)
    chains: int = 2
    draws: int = 500
    tune: int = 500
    target_accept: float = 0.9
    random_seed: int = 42
    use_in_ensemble: bool = True
    split_scope: str = "train"


@dataclass(frozen=True)
class NNConfig:
    seq_len: int = 10
    feature_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    pretrain_batch_size: int = 512
    pretrain_epochs: int = 30
    pretrain_lr: float = 0.001
    finetune_batch_size: int = 256
    finetune_epochs: int = 20
    finetune_lr: float = 0.0001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5
    device: str = "auto"
    num_workers: int = 0
    freeze_encoder_epochs: int = 0


@dataclass(frozen=True)
class AppConfig:
    elo: EloConfig
    features: FeatureConfig
    splits: SplitConfig
    gbm: GBMConfig
    simulation: SimulationConfig
    calibration: CalibrationConfig
    club: ClubConfig
    nn: NNConfig
    bayesian: BayesianConfig
    pipeline: PipelineConfig
    team_aliases: dict[str, str]


@dataclass(frozen=True)
class PipelineConfig:
    skip_club_pretrain: bool = False
    time_budget_hours: float = 0.0


@dataclass(frozen=True)
class TournamentConfig:
    year: int
    kickoff_date: date
    actual_champion: str | None = None
    groups: dict[str, list[str]] | None = None
    mode: str = "full"
    round_of_16: tuple[tuple[str, str], ...] = ()
    quarterfinal_pairings: tuple[tuple[int, int], ...] = ()

    @property
    def is_knockout_only(self) -> bool:
        return self.mode == "knockout_only"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_config(config_dir: Path | None = None) -> AppConfig:
    config_dir = config_dir or CONFIG_DIR
    defaults_path = config_dir / "defaults.yaml"
    aliases_path = config_dir / "team_aliases.yaml"

    with defaults_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    with aliases_path.open(encoding="utf-8") as f:
        aliases_raw: dict[str, Any] = yaml.safe_load(f)

    elo_raw = raw["elo"]
    feat_raw = raw["features"]
    split_raw = raw["splits"]
    gbm_raw = raw.get("gbm", {})
    sim_raw = raw.get("simulation", {})
    cal_raw = raw.get("calibration", {})
    club_raw = raw.get("club", {})
    nn_raw = raw.get("nn", {})
    bayesian_raw = raw.get("bayesian", {})
    pipeline_raw = raw.get("pipeline", {})
    scaling_bounds_raw = cal_raw.get("scaling_bounds", [0.5, 2.0])
    club_leagues = club_raw.get("leagues", [])
    understat_leagues = club_raw.get(
        "understat_leagues",
        [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ],
    )
    understat_seasons = club_raw.get(
        "understat_seasons", ["1415", "1516", "1617", "1718", "1819"]
    )

    return AppConfig(
        elo=EloConfig(
            initial=float(elo_raw["initial"]),
            k_factor=float(elo_raw["k_factor"]),
            home_advantage=float(elo_raw["home_advantage"]),
        ),
        features=FeatureConfig(
            form_decay=float(feat_raw["form_decay"]),
            form_window=int(feat_raw["form_window"]),
            rolling_goals_window=int(feat_raw["rolling_goals_window"]),
            h2h_decay=float(feat_raw["h2h_decay"]),
        ),
        splits=SplitConfig(
            train_end=_parse_date(split_raw["train_end"]),
            val_end=_parse_date(split_raw["val_end"]),
        ),
        gbm=GBMConfig(
            num_boost_round=int(gbm_raw.get("num_boost_round", 500)),
            learning_rate=float(gbm_raw.get("learning_rate", 0.05)),
            num_leaves=int(gbm_raw.get("num_leaves", 31)),
            early_stopping_rounds=int(gbm_raw.get("early_stopping_rounds", 50)),
        ),
        simulation=SimulationConfig(
            n_sims=int(sim_raw.get("n_sims", 10000)),
            max_goals=int(sim_raw.get("max_goals", 10)),
            dixon_coles_rho=float(sim_raw.get("dixon_coles_rho", 0.0)),
        ),
        calibration=CalibrationConfig(
            rho_min=float(cal_raw.get("rho_min", -0.2)),
            rho_max=float(cal_raw.get("rho_max", 0.1)),
            scaling_bounds=(
                float(scaling_bounds_raw[0]),
                float(scaling_bounds_raw[1]),
            ),
            min_ensemble_weight=float(cal_raw.get("min_ensemble_weight", 0.10)),
        ),
        club=ClubConfig(
            data_dir=str(club_raw.get("data_dir", "data/raw/club")),
            min_date=_parse_date(club_raw.get("min_date", "2014-01-01")),
            leagues=tuple(str(x) for x in club_leagues),
            forward_fill_within_league=bool(
                club_raw.get("forward_fill_within_league", True)
            ),
            understat_leagues=tuple(str(x) for x in understat_leagues),
            understat_seasons=tuple(str(x) for x in understat_seasons),
        ),
        nn=NNConfig(
            seq_len=int(nn_raw.get("seq_len", 10)),
            feature_dim=int(nn_raw.get("feature_dim", 10)),
            hidden_dim=int(nn_raw.get("hidden_dim", 64)),
            num_layers=int(nn_raw.get("num_layers", 1)),
            dropout=float(nn_raw.get("dropout", 0.1)),
            pretrain_batch_size=int(nn_raw.get("pretrain_batch_size", 512)),
            pretrain_epochs=int(nn_raw.get("pretrain_epochs", 30)),
            pretrain_lr=float(nn_raw.get("pretrain_lr", 0.001)),
            finetune_batch_size=int(nn_raw.get("finetune_batch_size", 256)),
            finetune_epochs=int(nn_raw.get("finetune_epochs", 20)),
            finetune_lr=float(nn_raw.get("finetune_lr", 0.0001)),
            weight_decay=float(nn_raw.get("weight_decay", 0.0001)),
            early_stopping_patience=int(
                nn_raw.get("early_stopping_patience", 5)
            ),
            device=str(nn_raw.get("device", "auto")),
            num_workers=int(nn_raw.get("num_workers", 0)),
            freeze_encoder_epochs=int(nn_raw.get("freeze_encoder_epochs", 0)),
        ),
        bayesian=BayesianConfig(
            min_date=_parse_date(bayesian_raw.get("min_date", "2000-01-01")),
            chains=int(bayesian_raw.get("chains", 2)),
            draws=int(bayesian_raw.get("draws", 500)),
            tune=int(bayesian_raw.get("tune", 500)),
            target_accept=float(bayesian_raw.get("target_accept", 0.9)),
            random_seed=int(bayesian_raw.get("random_seed", 42)),
            use_in_ensemble=bool(bayesian_raw.get("use_in_ensemble", True)),
            split_scope=str(bayesian_raw.get("split_scope", "train")),
        ),
        pipeline=PipelineConfig(
            skip_club_pretrain=bool(pipeline_raw.get("skip_club_pretrain", False)),
            time_budget_hours=float(pipeline_raw.get("time_budget_hours", 0.0)),
        ),
        team_aliases=dict(aliases_raw.get("aliases", {})),
    )


def load_tournament_config(path: Path) -> TournamentConfig:
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    mode = str(raw.get("mode", "full"))
    groups_raw = raw.get("groups")
    groups = {k: list(v) for k, v in groups_raw.items()} if groups_raw else None
    r16_raw = raw.get("round_of_16", [])
    qf_raw = raw.get("quarterfinal_pairings", [])
    actual = raw.get("actual_champion")
    return TournamentConfig(
        year=int(raw["year"]),
        kickoff_date=_parse_date(raw["kickoff_date"]),
        actual_champion=str(actual) if actual is not None else None,
        groups=groups,
        mode=mode,
        round_of_16=tuple(tuple(pair) for pair in r16_raw),
        quarterfinal_pairings=tuple(tuple(pair) for pair in qf_raw),
    )


def load_profile(profile_name: str, config_dir: Path | None = None) -> AppConfig:
    """Load defaults merged with a named profile overlay."""
    config_dir = config_dir or CONFIG_DIR
    base = load_config(config_dir)
    profile_path = config_dir / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with profile_path.open(encoding="utf-8") as f:
        overlay: dict[str, Any] = yaml.safe_load(f) or {}

    merged: dict[str, Any] = {
        "elo": {
            "initial": base.elo.initial,
            "k_factor": base.elo.k_factor,
            "home_advantage": base.elo.home_advantage,
        },
        "features": {
            "form_decay": base.features.form_decay,
            "form_window": base.features.form_window,
            "rolling_goals_window": base.features.rolling_goals_window,
            "h2h_decay": base.features.h2h_decay,
        },
        "splits": {
            "train_end": base.splits.train_end.isoformat(),
            "val_end": base.splits.val_end.isoformat(),
        },
        "gbm": {
            "num_boost_round": base.gbm.num_boost_round,
            "learning_rate": base.gbm.learning_rate,
            "num_leaves": base.gbm.num_leaves,
            "early_stopping_rounds": base.gbm.early_stopping_rounds,
        },
        "simulation": {
            "n_sims": base.simulation.n_sims,
            "max_goals": base.simulation.max_goals,
            "dixon_coles_rho": base.simulation.dixon_coles_rho,
        },
        "calibration": {
            "rho_min": base.calibration.rho_min,
            "rho_max": base.calibration.rho_max,
            "scaling_bounds": list(base.calibration.scaling_bounds),
            "min_ensemble_weight": base.calibration.min_ensemble_weight,
        },
        "club": {
            "data_dir": base.club.data_dir,
            "min_date": base.club.min_date.isoformat(),
            "leagues": list(base.club.leagues),
            "forward_fill_within_league": base.club.forward_fill_within_league,
            "understat_leagues": list(base.club.understat_leagues),
            "understat_seasons": list(base.club.understat_seasons),
        },
        "nn": {
            "seq_len": base.nn.seq_len,
            "feature_dim": base.nn.feature_dim,
            "hidden_dim": base.nn.hidden_dim,
            "num_layers": base.nn.num_layers,
            "dropout": base.nn.dropout,
            "pretrain_batch_size": base.nn.pretrain_batch_size,
            "pretrain_epochs": base.nn.pretrain_epochs,
            "pretrain_lr": base.nn.pretrain_lr,
            "finetune_batch_size": base.nn.finetune_batch_size,
            "finetune_epochs": base.nn.finetune_epochs,
            "finetune_lr": base.nn.finetune_lr,
            "weight_decay": base.nn.weight_decay,
            "early_stopping_patience": base.nn.early_stopping_patience,
            "device": base.nn.device,
            "num_workers": base.nn.num_workers,
            "freeze_encoder_epochs": base.nn.freeze_encoder_epochs,
        },
        "bayesian": {
            "min_date": base.bayesian.min_date.isoformat(),
            "chains": base.bayesian.chains,
            "draws": base.bayesian.draws,
            "tune": base.bayesian.tune,
            "target_accept": base.bayesian.target_accept,
            "random_seed": base.bayesian.random_seed,
            "use_in_ensemble": base.bayesian.use_in_ensemble,
            "split_scope": base.bayesian.split_scope,
        },
        "pipeline": {
            "skip_club_pretrain": base.pipeline.skip_club_pretrain,
            "time_budget_hours": base.pipeline.time_budget_hours,
        },
    }

    for section, values in overlay.items():
        if section not in merged:
            merged[section] = values
        elif isinstance(values, dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values

    defaults_path = config_dir / "defaults.yaml"
    aliases_path = config_dir / "team_aliases.yaml"
    with defaults_path.open(encoding="utf-8") as f:
        defaults_raw = yaml.safe_load(f)
    with aliases_path.open(encoding="utf-8") as f:
        aliases_raw = yaml.safe_load(f)

    for section in ("elo", "features", "splits", "gbm", "simulation", "calibration", "club", "nn", "bayesian", "pipeline"):
        if section in merged:
            defaults_raw[section] = merged[section]

    temp_path = config_dir / ".merged_profile.yaml"
    with temp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(defaults_raw, f)

    try:
        return _load_config_from_raw(defaults_raw, aliases_raw)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_config_from_raw(raw: dict[str, Any], aliases_raw: dict[str, Any]) -> AppConfig:
    elo_raw = raw["elo"]
    feat_raw = raw["features"]
    split_raw = raw["splits"]
    gbm_raw = raw.get("gbm", {})
    sim_raw = raw.get("simulation", {})
    cal_raw = raw.get("calibration", {})
    club_raw = raw.get("club", {})
    nn_raw = raw.get("nn", {})
    bayesian_raw = raw.get("bayesian", {})
    pipeline_raw = raw.get("pipeline", {})
    scaling_bounds_raw = cal_raw.get("scaling_bounds", [0.5, 2.0])
    club_leagues = club_raw.get("leagues", [])
    understat_leagues = club_raw.get(
        "understat_leagues",
        [
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
        ],
    )
    understat_seasons = club_raw.get(
        "understat_seasons", ["1415", "1516", "1617", "1718", "1819"]
    )

    return AppConfig(
        elo=EloConfig(
            initial=float(elo_raw["initial"]),
            k_factor=float(elo_raw["k_factor"]),
            home_advantage=float(elo_raw["home_advantage"]),
        ),
        features=FeatureConfig(
            form_decay=float(feat_raw["form_decay"]),
            form_window=int(feat_raw["form_window"]),
            rolling_goals_window=int(feat_raw["rolling_goals_window"]),
            h2h_decay=float(feat_raw["h2h_decay"]),
        ),
        splits=SplitConfig(
            train_end=_parse_date(split_raw["train_end"]),
            val_end=_parse_date(split_raw["val_end"]),
        ),
        gbm=GBMConfig(
            num_boost_round=int(gbm_raw.get("num_boost_round", 500)),
            learning_rate=float(gbm_raw.get("learning_rate", 0.05)),
            num_leaves=int(gbm_raw.get("num_leaves", 31)),
            early_stopping_rounds=int(gbm_raw.get("early_stopping_rounds", 50)),
        ),
        simulation=SimulationConfig(
            n_sims=int(sim_raw.get("n_sims", 10000)),
            max_goals=int(sim_raw.get("max_goals", 10)),
            dixon_coles_rho=float(sim_raw.get("dixon_coles_rho", 0.0)),
        ),
        calibration=CalibrationConfig(
            rho_min=float(cal_raw.get("rho_min", -0.2)),
            rho_max=float(cal_raw.get("rho_max", 0.1)),
            scaling_bounds=(
                float(scaling_bounds_raw[0]),
                float(scaling_bounds_raw[1]),
            ),
            min_ensemble_weight=float(cal_raw.get("min_ensemble_weight", 0.10)),
        ),
        club=ClubConfig(
            data_dir=str(club_raw.get("data_dir", "data/raw/club")),
            min_date=_parse_date(club_raw.get("min_date", "2014-01-01")),
            leagues=tuple(str(x) for x in club_leagues),
            forward_fill_within_league=bool(
                club_raw.get("forward_fill_within_league", True)
            ),
            understat_leagues=tuple(str(x) for x in understat_leagues),
            understat_seasons=tuple(str(x) for x in understat_seasons),
        ),
        nn=NNConfig(
            seq_len=int(nn_raw.get("seq_len", 10)),
            feature_dim=int(nn_raw.get("feature_dim", 10)),
            hidden_dim=int(nn_raw.get("hidden_dim", 64)),
            num_layers=int(nn_raw.get("num_layers", 1)),
            dropout=float(nn_raw.get("dropout", 0.1)),
            pretrain_batch_size=int(nn_raw.get("pretrain_batch_size", 512)),
            pretrain_epochs=int(nn_raw.get("pretrain_epochs", 30)),
            pretrain_lr=float(nn_raw.get("pretrain_lr", 0.001)),
            finetune_batch_size=int(nn_raw.get("finetune_batch_size", 256)),
            finetune_epochs=int(nn_raw.get("finetune_epochs", 20)),
            finetune_lr=float(nn_raw.get("finetune_lr", 0.0001)),
            weight_decay=float(nn_raw.get("weight_decay", 0.0001)),
            early_stopping_patience=int(
                nn_raw.get("early_stopping_patience", 5)
            ),
            device=str(nn_raw.get("device", "auto")),
            num_workers=int(nn_raw.get("num_workers", 0)),
            freeze_encoder_epochs=int(nn_raw.get("freeze_encoder_epochs", 0)),
        ),
        bayesian=BayesianConfig(
            min_date=_parse_date(bayesian_raw.get("min_date", "2000-01-01")),
            chains=int(bayesian_raw.get("chains", 2)),
            draws=int(bayesian_raw.get("draws", 500)),
            tune=int(bayesian_raw.get("tune", 500)),
            target_accept=float(bayesian_raw.get("target_accept", 0.9)),
            random_seed=int(bayesian_raw.get("random_seed", 42)),
            use_in_ensemble=bool(bayesian_raw.get("use_in_ensemble", True)),
            split_scope=str(bayesian_raw.get("split_scope", "train")),
        ),
        pipeline=PipelineConfig(
            skip_club_pretrain=bool(pipeline_raw.get("skip_club_pretrain", False)),
            time_budget_hours=float(pipeline_raw.get("time_budget_hours", 0.0)),
        ),
        team_aliases=dict(aliases_raw.get("aliases", {})),
    )


def merge_former_name_aliases(
    config: AppConfig, former_names_path: Path
) -> AppConfig:
    """Add former→current mappings from martj42 former_names.csv."""
    import pandas as pd

    if not former_names_path.exists():
        return config

    df = pd.read_csv(former_names_path)
    aliases = dict(config.team_aliases)
    for _, row in df.iterrows():
        current = str(row["current"]).strip()
        former = str(row["former"]).strip()
        if former and former != current:
            aliases[former] = current

    return AppConfig(
        elo=config.elo,
        features=config.features,
        splits=config.splits,
        gbm=config.gbm,
        simulation=config.simulation,
        calibration=config.calibration,
        club=config.club,
        nn=config.nn,
        bayesian=config.bayesian,
        pipeline=config.pipeline,
        team_aliases=aliases,
    )
