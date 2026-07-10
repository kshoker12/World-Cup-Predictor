#!/usr/bin/env python3
"""Score completed knockout round predictions against actual results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUND_LABELS = {
    "round_of_16": "Round of 16",
    "quarter_finals": "Quarter-finals",
    "semi_finals": "Semi-finals",
    "final": "Final",
}

# Actual R16 results (home, away, winner, score)
R16_ACTUAL: list[dict[str, str]] = [
    {"home": "Paraguay", "away": "France", "winner": "France", "score": "0-1"},
    {"home": "Canada", "away": "Morocco", "winner": "Morocco", "score": "0-3"},
    {"home": "Brazil", "away": "Norway", "winner": "Norway", "score": "1-2"},
    {"home": "Mexico", "away": "England", "winner": "England", "score": "2-3"},
    {"home": "Spain", "away": "Portugal", "winner": "Spain", "score": "1-0"},
    {"home": "United States", "away": "Belgium", "winner": "Belgium", "score": "1-4"},
    {"home": "Argentina", "away": "Egypt", "winner": "Argentina", "score": "3-2"},
    {"home": "Switzerland", "away": "Colombia", "winner": "Switzerland", "score": "1-0"},
]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _match_lookup(match_win_probs: list[dict], rnd: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in match_win_probs:
        if row.get("round") != rnd:
            continue
        key = (row["home"], row["away"])
        out[key] = row
    return out


def score_round(
    forecast: dict,
    *,
    rnd: str,
    actual_results: list[dict[str, str]],
    forecast_file: str,
    n_sims: int | None = None,
    completed_at: str = "2026-07-07",
) -> dict:
    lookup = _match_lookup(forecast.get("match_win_probs", []), rnd)
    scored: list[dict] = []
    correct = 0

    for actual in actual_results:
        home, away = actual["home"], actual["away"]
        row = lookup.get((home, away))
        if row is None:
            raise ValueError(f"No forecast for {home} vs {away} in round {rnd}")

        p_home = float(row["p_home_win"])
        p_away = float(row["p_away_win"])
        predicted = home if p_home >= p_away else away
        is_correct = predicted == actual["winner"]
        if is_correct:
            correct += 1

        scored.append(
            {
                "home": home,
                "away": away,
                "winner": actual["winner"],
                "score": actual["score"],
                "predicted_winner": predicted,
                "p_home_win": p_home,
                "p_away_win": p_away,
                "correct": is_correct,
            }
        )

    total = len(actual_results)
    return {
        "round": rnd,
        "label": ROUND_LABELS.get(rnd, rnd),
        "forecast_file": forecast_file,
        "n_sims": n_sims if n_sims is not None else forecast.get("n_sims"),
        "completed_at": completed_at,
        "results": scored,
        "summary": {"correct": correct, "total": total},
    }


def build_round_history(
    *,
    forecast_path: Path,
    rnd: str = "round_of_16",
    actual_results: list[dict[str, str]] | None = None,
    active_round: str = "quarter_finals",
    active_forecast: str = "wc2026_forecast_qf.json",
    history_forecast_relpath: str | None = None,
) -> dict:
    forecast = _load(forecast_path)
    rel = history_forecast_relpath or forecast_path.name
    if history_forecast_relpath and not history_forecast_relpath.startswith("history/"):
        rel = f"history/{history_forecast_relpath}"

    actuals = actual_results if actual_results is not None else R16_ACTUAL
    if rnd == "round_of_16" and actual_results is None:
        actuals = R16_ACTUAL

    round_entry = score_round(
        forecast,
        rnd=rnd,
        actual_results=actuals,
        forecast_file=rel,
    )
    return {
        "rounds": [round_entry],
        "active_round": active_round,
        "active_forecast": active_forecast,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score forecast picks vs actual results")
    parser.add_argument("forecast", type=Path, help="Forecast JSON (pre-round archive)")
    parser.add_argument("--round", default="round_of_16")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "data" / "round_history.json",
    )
    parser.add_argument(
        "--history-relpath",
        default="history/forecast_pre_r16.json",
        help="Relative path stored in round_history for the forecast file",
    )
    parser.add_argument("--active-round", default="quarter_finals")
    parser.add_argument("--active-forecast", default="wc2026_forecast_qf.json")
    args = parser.parse_args()

    if not args.forecast.exists():
        print(f"ERROR: Missing {args.forecast}", file=sys.stderr)
        return 1

    payload = build_round_history(
        forecast_path=args.forecast,
        rnd=args.round,
        active_round=args.active_round,
        active_forecast=args.active_forecast,
        history_forecast_relpath=args.history_relpath,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary = payload["rounds"][0]["summary"]
    print(f"Scored {args.round}: {summary['correct']}/{summary['total']} correct")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
