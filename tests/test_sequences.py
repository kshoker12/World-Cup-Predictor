import numpy as np
import pandas as pd

from worldcup_predictor.models.sequences import (
    SEQ_FEATURE_DIM,
    TeamSequenceState,
    build_club_sequences,
    build_international_sequences,
    make_timestep_vector,
)


def test_padding_and_mask(default_config):
    state = TeamSequenceState(seq_len=3)
    snap = state.snapshot()
    assert snap.shape == (3, SEQ_FEATURE_DIM)
    assert snap.sum() == 0.0


def test_international_mask_zero(default_config):
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2015-01-01").date(),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "tournament": "Friendly",
                "neutral": True,
            }
        ]
    )
    home_seq, away_seq = build_international_sequences(matches, default_config)
    assert home_seq[0, -1, -1] == 0.0


def test_club_sequences_no_leakage(default_config):
    matches = pd.DataFrame(
        [
            {
                "date": "2018-01-01",
                "home_team": "a",
                "away_team": "b",
                "home_goals": 2,
                "away_goals": 1,
                "home_xg": 1.5,
                "away_xg": 0.8,
                "home_shots": 10,
                "away_shots": 8,
                "home_ppda": 8,
                "away_ppda": 9,
                "league": "EPL",
                "source": "understat",
            },
            {
                "date": "2018-01-08",
                "home_team": "c",
                "away_team": "d",
                "home_goals": 0,
                "away_goals": 0,
                "home_xg": 0.6,
                "away_xg": 0.5,
                "home_shots": 6,
                "away_shots": 5,
                "home_ppda": 7,
                "away_ppda": 8,
                "league": "EPL",
                "source": "understat",
            },
        ]
    )
    home_seq, away_seq, y_h, y_a = build_club_sequences(
        matches, seq_len=3, elo_config=default_config.elo
    )
    assert home_seq[1].sum() == 0.0
    assert y_h[0] == 2
    assert y_a[1] == 0
