"""Knockout bracket construction and advancement."""

from __future__ import annotations

# Standard FIFA World Cup Round of 16 pairings (32-team format)
R16_PAIRINGS: list[tuple[str, str]] = [
    ("A1", "B2"),
    ("C1", "D2"),
    ("E1", "F2"),
    ("G1", "H2"),
    ("B1", "A2"),
    ("D1", "C2"),
    ("F1", "E2"),
    ("H1", "G2"),
]


def _resolve_seed(seed: str, ranked: dict[str, list[str]]) -> str:
    group = seed[0]
    position = int(seed[1]) - 1
    return ranked[group][position]


def build_round_of_16(ranked: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Return list of (home, away) team name pairs for Round of 16."""
    fixtures: list[tuple[str, str]] = []
    for home_seed, away_seed in R16_PAIRINGS:
        fixtures.append(
            (_resolve_seed(home_seed, ranked), _resolve_seed(away_seed, ranked))
        )
    return fixtures


def advance_winners(results: list[tuple[str, str, str]]) -> list[str]:
    """Given (home, away, winner) per match, return winners in order."""
    return [winner for _, _, winner in results]


def build_knockout_round(winners: list[str]) -> list[tuple[str, str]]:
    """Pair adjacent winners: w0 vs w1, w2 vs w3, ..."""
    fixtures: list[tuple[str, str]] = []
    for i in range(0, len(winners), 2):
        fixtures.append((winners[i], winners[i + 1]))
    return fixtures
