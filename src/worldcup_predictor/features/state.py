"""Rolling team and head-to-head state for feature computation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def _result_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


@dataclass
class TeamRollingState:
    form_window: int
    goals_window: int
    form_decay: float
    _matches: deque[tuple[int, int]] = field(default_factory=deque)

    def snapshot_form_points(self) -> float:
        total = 0.0
        for i, (gf, ga) in enumerate(reversed(list(self._matches)[-self.form_window :])):
            total += (self.form_decay**i) * _result_points(gf, ga)
        return total

    def snapshot_avg_goals_for(self) -> float:
        recent = list(self._matches)[-self.goals_window :]
        if not recent:
            return 0.0
        return sum(gf for gf, _ in recent) / len(recent)

    def snapshot_avg_goals_against(self) -> float:
        recent = list(self._matches)[-self.goals_window :]
        if not recent:
            return 0.0
        return sum(ga for _, ga in recent) / len(recent)

    def update(self, goals_for: int, goals_against: int) -> None:
        self._matches.append((goals_for, goals_against))
        max_len = max(self.form_window, self.goals_window)
        while len(self._matches) > max_len:
            self._matches.popleft()


@dataclass
class H2HState:
    h2h_decay: float
    # Each entry: (home_team, away_team, home_score, away_score) for a prior meeting
    _meetings: deque[tuple[str, str, int, int]] = field(default_factory=deque)

    def snapshot(self, current_home: str, current_away: str) -> float:
        pair = {current_home, current_away}
        relevant = [
            (h, a, hs, aws)
            for h, a, hs, aws in self._meetings
            if {h, a} == pair
        ]
        total = 0.0
        for i, (h, a, hs, aws) in enumerate(reversed(relevant)):
            if current_home == h and current_away == a:
                gd = hs - aws
            else:
                gd = aws - hs
            total += (self.h2h_decay**i) * gd
        return total

    def update(self, home_team: str, away_team: str, home_score: int, away_score: int) -> None:
        self._meetings.append((home_team, away_team, home_score, away_score))


def tournament_importance(tournament: str) -> int:
    t = tournament.lower()
    if "qualif" in t:
        return 2
    if "world cup" in t:
        return 3
    if "friendly" in t:
        return 1
    return 1
