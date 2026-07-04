import pandas as pd

from conftest import make_match, matches_df
from worldcup_predictor.features.pipeline import MatchPipeline


def _build_features(config, rows):
    return MatchPipeline(config).run(matches_df(rows))


def test_truncated_replay_identical(default_config):
    rows = [
        make_match("2020-01-01", "A", "B", 2, 1),
        make_match("2020-02-01", "C", "D", 0, 0),
        make_match("2020-03-01", "A", "C", 1, 0),
        make_match("2020-04-01", "B", "D", 3, 2),
    ]
    full = _build_features(default_config, rows)
    target_idx = 2
    truncated = _build_features(default_config, rows[: target_idx + 1])
    target_cols = [
        "elo_diff",
        "gf_last_5_diff",
        "ga_last_5_diff",
        "form_diff",
        "is_home",
        "is_neutral",
        "tournament_importance",
        "h2h_gd_weighted",
    ]
    for col in target_cols:
        assert full.iloc[target_idx][col] == truncated.iloc[target_idx][col]


def test_future_shuffle_invariant(default_config):
    rows = [
        make_match("2020-01-01", "A", "B", 2, 1),
        make_match("2020-02-01", "C", "D", 0, 0),
        make_match("2020-03-01", "A", "C", 1, 0),
    ]
    baseline = _build_features(default_config, rows)
    target = baseline.iloc[1]

    # Append future matches that should not affect row index 1
    extended = rows + [
        make_match("2021-01-01", "E", "F", 5, 0),
        make_match("2021-02-01", "C", "E", 2, 2),
    ]
    extended_out = _build_features(default_config, extended)
    assert extended_out.iloc[1]["elo_diff"] == target["elo_diff"]
    assert extended_out.iloc[1]["form_diff"] == target["form_diff"]


def test_same_date_global_sort_order(default_config):
    """Two matches same day: sort key (date, home_team, away_team)."""
    rows = [
        make_match("2020-01-01", "B", "C", 1, 0),
        make_match("2020-01-01", "A", "D", 2, 0),
    ]
    df = matches_df(rows)
    df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    out = MatchPipeline(default_config).run(df)
    # A plays first (home A), so B's match hasn't happened when A plays
    assert out.iloc[0]["home_team"] == "A"
    assert out.iloc[1]["home_team"] == "B"


def test_leakage_scores_not_in_features(default_config):
    rows = [make_match("2020-01-01", "A", "B", 5, 0)]
    out = _build_features(default_config, rows)
    feature_cols = [
        "elo_diff",
        "gf_last_5_diff",
        "ga_last_5_diff",
        "form_diff",
        "h2h_gd_weighted",
    ]
    for col in feature_cols:
        assert out.iloc[0][col] == 0.0  # no prior history uses current score
