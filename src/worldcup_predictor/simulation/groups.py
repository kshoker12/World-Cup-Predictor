"""Group stage round-robin and standings."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class TeamStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class GroupStandings:
    group_name: str
    teams: dict[str, TeamStanding] = field(default_factory=dict)

    def init_teams(self, team_names: list[str]) -> None:
        self.teams = {t: TeamStanding(team=t) for t in team_names}

    def record_result(
        self, home: str, away: str, home_goals: int, away_goals: int
    ) -> None:
        hs = self.teams[home]
        aws = self.teams[away]
        hs.played += 1
        aws.played += 1
        hs.goals_for += home_goals
        hs.goals_against += away_goals
        aws.goals_for += away_goals
        aws.goals_against += home_goals

        if home_goals > away_goals:
            hs.wins += 1
            aws.losses += 1
        elif home_goals < away_goals:
            aws.wins += 1
            hs.losses += 1
        else:
            hs.draws += 1
            aws.draws += 1

    def ranked(self) -> list[str]:
        """Rank by points, then GD, then goals for."""
        return sorted(
            self.teams.values(),
            key=lambda s: (s.points, s.goal_difference, s.goals_for),
            reverse=True,
        )


def group_fixtures(teams: list[str]) -> list[tuple[str, str]]:
    """Round-robin home/away pairs for 4 teams (6 matches)."""
    fixtures: list[tuple[str, str]] = []
    for home, away in combinations(teams, 2):
        fixtures.append((home, away))
    return fixtures


def rank_groups(group_teams: dict[str, list[str]], results: list[tuple]) -> dict[str, list[str]]:
    """
    Record group results and return {group: [1st, 2nd, 3rd, 4th]}.
    results: list of (group_name, home, away, home_goals, away_goals)
    """
    standings: dict[str, GroupStandings] = {}
    for gname, teams in group_teams.items():
        gs = GroupStandings(group_name=gname)
        gs.init_teams(teams)
        standings[gname] = gs

    for gname, home, away, hg, ag in results:
        standings[gname].record_result(home, away, hg, ag)

    return {g: [s.team for s in standings[g].ranked()] for g in group_teams}
