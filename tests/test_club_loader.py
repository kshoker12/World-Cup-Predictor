from pathlib import Path

import pandas as pd

from worldcup_predictor.data.club_loader import load_soccer_dataset
from worldcup_predictor.data.club_merge import merge_club_sources

FIXTURES = Path(__file__).parent / "fixtures"


def test_merge_prefers_understat_on_overlap(tmp_path):
    understat = pd.read_csv(FIXTURES / "club_understat.csv")
    soccer_dir = tmp_path / "club"
    soccer_dir.mkdir()
    pd.read_csv(FIXTURES / "club_fixtures.csv").to_csv(
        soccer_dir / "fixtures.csv", index=False
    )
    pd.read_csv(FIXTURES / "club_match_stats.csv").to_csv(
        soccer_dir / "match_stats.csv", index=False
    )
    soccer = load_soccer_dataset(soccer_dir)
    merged = merge_club_sources(understat, soccer)
    assert len(merged) == 3
    overlap = merged[
        (merged["home_team"] == "alpha") & (merged["away_team"] == "beta")
    ]
    assert overlap.iloc[0]["source"] == "understat"
