from conftest import make_match, matches_df
from worldcup_predictor.features.pipeline import MatchPipeline
from worldcup_predictor.features.state import TeamRollingState, tournament_importance


def test_tournament_importance_mapping():
    assert tournament_importance("FIFA World Cup") == 3
    assert tournament_importance("World Cup qualification") == 2
    assert tournament_importance("Friendly") == 1
    assert tournament_importance("Copa América") == 1


def test_first_match_features(default_config):
    rows = [make_match("2020-01-01", "A", "B", 2, 1)]
    out = MatchPipeline(default_config).run(matches_df(rows))
    row = out.iloc[0]
    assert row["elo_diff"] == 0.0
    assert row["gf_last_5_diff"] == 0.0
    assert row["ga_last_5_diff"] == 0.0
    assert row["form_diff"] == 0.0
    assert row["h2h_gd_weighted"] == 0.0
    assert row["is_home"] == 1
    assert row["is_neutral"] == 0


def test_neutral_match_features(default_config):
    rows = [make_match("2020-01-01", "A", "B", 1, 1, neutral=True)]
    out = MatchPipeline(default_config).run(matches_df(rows))
    assert out.iloc[0]["is_home"] == 0
    assert out.iloc[0]["is_neutral"] == 1


def test_form_diff_after_sequence(default_config):
    # A wins twice (6 pts weighted), B draws once (1 pt)
    # A match 1: A 2-0 B -> A form before m1 = 0
    # B match 2: B 1-1 C -- skip, test A vs B twice
    rows = [
        make_match("2020-01-01", "A", "B", 2, 0),
        make_match("2020-02-01", "A", "B", 1, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    second = out.iloc[1]
    # Before match 2: A form = 3, B form = 0 -> form_diff = 3
    assert second["form_diff"] == 3.0


def test_h2h_weighted(default_config):
    rows = [
        make_match("2020-01-01", "A", "B", 2, 0),
        make_match("2020-02-01", "B", "A", 1, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    # Match 2 home=B: prior H2H is A 2-0 B -> gd from B perspective = -2
    assert out.iloc[1]["h2h_gd_weighted"] == -2.0


def test_gf_ga_last_5(default_config):
    rows = [
        make_match("2020-01-01", "A", "X", 4, 0),
        make_match("2020-02-01", "A", "X", 2, 0),
        make_match("2020-03-01", "A", "B", 0, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    third = out.iloc[2]
    # A avg gf = (4+2)/2 = 3, B avg gf = 0
    assert third["gf_last_5_diff"] == 3.0


def test_team_rolling_state_form():
    state = TeamRollingState(form_window=10, goals_window=5, form_decay=0.9)
    state.update(2, 0)  # win = 3 pts, i=0
    state.update(1, 1)  # draw = 1 pt, i=0 most recent
    # form = 1*1 + 0.9*3 = 3.7
    assert abs(state.snapshot_form_points() - 3.7) < 1e-9
