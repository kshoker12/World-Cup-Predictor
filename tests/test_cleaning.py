from datetime import date

import pandas as pd

from worldcup_predictor.data.cleaning import clean_matches


def test_drops_invalid_rows():
    df = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "A",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
            {
                "date": "bad-date",
                "home_team": "B",
                "away_team": "C",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
            {
                "date": "2020-01-02",
                "home_team": "B",
                "away_team": "C",
                "home_score": -1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
            {
                "date": "2020-01-03",
                "home_team": "B",
                "away_team": "C",
                "home_score": 2,
                "away_score": 1,
                "tournament": "Friendly",
                "neutral": False,
            },
        ]
    )
    cleaned = clean_matches(df, {})
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["home_team"] == "B"


def test_deduplicates():
    row = {
        "date": "2020-01-01",
        "home_team": "A",
        "away_team": "B",
        "home_score": 1,
        "away_score": 0,
        "tournament": "Friendly",
        "neutral": False,
    }
    df = pd.DataFrame([row, row])
    cleaned = clean_matches(df, {})
    assert len(cleaned) == 1


def test_team_alias_normalization():
    df = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "Czech Republic",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            }
        ]
    )
    cleaned = clean_matches(df, {"Czech Republic": "Czechia"})
    assert cleaned.iloc[0]["home_team"] == "Czechia"


def test_sort_order():
    df = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "home_team": "B",
                "away_team": "C",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 0,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": False,
            },
        ]
    )
    cleaned = clean_matches(df, {})
    assert cleaned.iloc[0]["date"] == date(2020, 1, 1)


def test_neutral_parsing():
    df = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": "TRUE",
            }
        ]
    )
    cleaned = clean_matches(df, {})
    assert bool(cleaned.iloc[0]["neutral"]) is True
