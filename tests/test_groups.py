from worldcup_predictor.simulation.groups import GroupStandings, group_fixtures, rank_groups


def test_four_team_fixtures():
    teams = ["A", "B", "C", "D"]
    assert len(group_fixtures(teams)) == 6


def test_standings_ranking():
    gs = GroupStandings("A")
    gs.init_teams(["A", "B", "C", "D"])
    gs.record_result("A", "B", 3, 0)
    gs.record_result("A", "C", 1, 1)
    gs.record_result("B", "C", 0, 1)
    ranked = gs.ranked()
    assert ranked[0].team == "A"
    assert ranked[0].points >= ranked[1].points


def test_tiebreaker_gd():
    results = [
        ("G", "A", "B", 1, 0),
        ("G", "C", "D", 2, 2),
        ("G", "A", "C", 3, 0),
        ("G", "B", "D", 1, 1),
        ("G", "A", "D", 0, 0),
        ("G", "B", "C", 0, 2),
    ]
    ranked = rank_groups({"G": ["A", "B", "C", "D"]}, results)
    assert ranked["G"][0] == "A"
