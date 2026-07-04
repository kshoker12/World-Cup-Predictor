from worldcup_predictor.simulation.bracket import R16_PAIRINGS, build_round_of_16


def test_r16_pairings_count():
    assert len(R16_PAIRINGS) == 8


def test_r16_first_pairing():
    ranked = {
        g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"]
        for g in "ABCDEFGH"
    }
    fixtures = build_round_of_16(ranked)
    assert fixtures[0] == ("A1", "B2")
    assert fixtures[4] == ("B1", "A2")
