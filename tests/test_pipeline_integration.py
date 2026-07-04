from datetime import date

from conftest import make_match, matches_df
from worldcup_predictor.features.pipeline import MatchPipeline
from worldcup_predictor.ratings.elo import expected_score, update_rating
from worldcup_predictor.splits import assign_split


def test_split_boundaries(default_config):
    assert assign_split(date(2009, 12, 31), default_config.splits) == "train"
    assert assign_split(date(2010, 1, 1), default_config.splits) == "val"
    assert assign_split(date(2017, 12, 31), default_config.splits) == "val"
    assert assign_split(date(2018, 1, 1), default_config.splits) == "test"


def test_row_count_parity(default_config):
    rows = [make_match(f"2020-0{i}-01", "A", "B", 1, 0) for i in range(1, 6)]
    matches = matches_df(rows)
    out = MatchPipeline(default_config).run(matches)
    assert len(out) == len(matches)


def test_elo_after_known_sequence(default_config):
    # Neutral 1500 vs 1500, home wins 1-0
    rows = [make_match("2020-01-01", "A", "B", 1, 0, neutral=True)]
    pipeline = MatchPipeline(default_config)
    pipeline.run(matches_df(rows))
    ratings = pipeline.elo_ratings()

    exp_home = expected_score(1500, 1500)
    expected_a = update_rating(1500, exp_home, 1.0, 20)
    expected_b = update_rating(1500, 1 - exp_home, 0.0, 20)
    assert abs(ratings["A"] - expected_a) < 1e-9
    assert abs(ratings["B"] - expected_b) < 1e-9


def test_h2h_three_meetings(default_config):
    rows = [
        make_match("2020-01-01", "A", "B", 2, 0),
        make_match("2020-02-01", "B", "A", 1, 0),
        make_match("2020-03-01", "A", "B", 0, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    third = out.iloc[2]
    # Prior H2H from A home: +2, then B home A lost 1-0 -> -1
    # h2h from A home persp: gd sequence [+2, -1], i=0 most recent is -1
    # weighted = -1 + 0.85*2 = 0.7
    assert abs(third["h2h_gd_weighted"] - 0.7) < 1e-9


def test_form_streak(default_config):
    rows = [
        make_match("2020-01-01", "A", "X", 1, 0),
        make_match("2020-02-01", "A", "X", 1, 0),
        make_match("2020-03-01", "A", "B", 0, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    third = out.iloc[2]
    # A form = 3 + 0.9*3 = 5.7, B form = 0
    assert abs(third["form_diff"] - 5.7) < 1e-9


def test_split_labels_in_output(default_config):
    rows = [
        make_match("2009-06-01", "A", "B", 1, 0),
        make_match("2015-06-01", "A", "B", 1, 0),
        make_match("2019-06-01", "A", "B", 1, 0),
    ]
    out = MatchPipeline(default_config).run(matches_df(rows))
    assert out.iloc[0]["split"] == "train"
    assert out.iloc[1]["split"] == "val"
    assert out.iloc[2]["split"] == "test"
