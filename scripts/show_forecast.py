#!/usr/bin/env python3
"""Pretty-print and HTML-export WC 2026 forecast results."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUND_LABELS = {
    "round_of_16": "Round of 16",
    "quarter_finals": "Quarter-finals",
    "semi_finals": "Semi-finals",
    "final": "Final",
    "third_place": "Third-place match",
}

ADVANCEMENT_COLUMNS = (
    ("champion", "Win it all"),
    ("final", "Make the final"),
    ("semi_finals", "Make the semis"),
    ("quarter_finals", "Make the quarters"),
)

PATH_ROUNDS_R16 = ("round_of_16", "quarter_finals", "semi_finals")
PATH_ROUNDS_QF = ("quarter_finals", "semi_finals")
PATH_ROUNDS_SF = ("semi_finals",)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _round_matches(match_win_probs: list[dict], rnd: str) -> list[dict]:
    rows = [r for r in match_win_probs if r.get("round") == rnd]
    return sorted(rows, key=lambda r: (r["home"], r["away"]))


def _match_prob(
    match_win_probs: list[dict],
    home: str,
    away: str,
    rnd: str,
) -> tuple[float | None, float | None]:
    for row in match_win_probs:
        if row.get("round") != rnd:
            continue
        if row["home"] == home and row["away"] == away:
            return float(row["p_home_win"]), float(row["p_away_win"])
    return None, None


def _active_round(data: dict, history: dict | None) -> str:
    sim = data.get("simulation_round")
    if sim in ROUND_LABELS:
        return str(sim)
    if history and history.get("active_round") in ROUND_LABELS:
        return str(history["active_round"])
    return "round_of_16"


def _is_qf_forecast(data: dict, history: dict | None) -> bool:
    return _active_round(data, history) == "quarter_finals"


def _is_sf_forecast(data: dict, history: dict | None) -> bool:
    return _active_round(data, history) == "semi_finals"


def _advancement_columns(data: dict, history: dict | None) -> tuple[tuple[str, str], ...]:
    active = _active_round(data, history)
    if active == "final":
        return (("champion", "Win it all"),)
    if active == "semi_finals":
        return tuple(c for c in ADVANCEMENT_COLUMNS if c[0] in ("champion", "final"))
    if active == "quarter_finals":
        return tuple(c for c in ADVANCEMENT_COLUMNS if c[0] != "quarter_finals")
    return ADVANCEMENT_COLUMNS


def _path_rounds(data: dict, history: dict | None) -> tuple[str, ...]:
    active = _active_round(data, history)
    if active == "final":
        return ()
    if active == "semi_finals":
        return PATH_ROUNDS_SF
    if active == "quarter_finals":
        return PATH_ROUNDS_QF
    return PATH_ROUNDS_R16


def _meta_line(data: dict, history: dict | None) -> str:
    n_sims = data.get("n_sims", 0)
    active = _active_round(data, history)
    if data.get("completed"):
        champion = data.get("champion") or (history or {}).get("champion", "?")
        return f"Tournament complete · Champion: {champion}"
    if data.get("pending"):
        return "Final & third-place fixtures set · Forecast pending"
    if active == "final":
        return f"Based on {n_sims:,} simulations · Final & third-place match"
    if active == "semi_finals":
        return f"Based on {n_sims:,} simulations · Semi-finals"
    if active == "quarter_finals":
        return f"Based on {n_sims:,} simulations · Quarter-finals"
    return f"Based on {n_sims:,} simulated tournaments"


def print_report(data: dict, history: dict | None = None) -> None:
    n_sims = data.get("n_sims", 0)
    print("=" * 72)
    print(f"WC 2026 FORECAST — {n_sims:,} simulations")
    print("=" * 72)

    print("\n--- Champion probabilities (top 10) ---")
    champ = data.get("champion_probs", {})
    for team, prob in sorted(champ.items(), key=lambda x: -x[1])[:10]:
        bar = "#" * int(prob * 40)
        print(f"  {team:20s} {prob:6.2%}  {bar}")

    adv = data.get("advancement_probs", {})
    cols = _advancement_columns(data, history)
    print("\n--- Tournament outlook (top 10) ---")
    header = f"  {'Team':20s}" + "".join(f" {label:>8s}" for _, label in cols)
    print(header)
    for team in sorted(champ, key=lambda t: -champ[t])[:10]:
        row = adv.get(team, {})
        parts = f"  {team:20s}"
        for key, _ in cols:
            parts += f" {row.get(key, 0):8.1%}"
        print(parts)

    if history:
        for rnd_entry in reversed(history.get("rounds", [])):
            summary = rnd_entry.get("summary", {})
            print(
                f"\n--- {rnd_entry.get('label', rnd_entry.get('round'))} "
                f"({summary.get('correct', 0)}/{summary.get('total', 0)} correct) ---"
            )
            for m in rnd_entry.get("results", []):
                mark = "ok" if m.get("correct") else "miss"
                print(
                    f"  {m['home']:18s} vs {m['away']:18s}  "
                    f"pick: {m['predicted_winner']:18s}  actual: {m['winner']:18s}  [{mark}]"
                )

    active_round = _active_round(data, history)
    print(f"\n--- {ROUND_LABELS[active_round]} ---")
    for m in _round_matches(data.get("match_win_probs", []), active_round):
        home, away = m["home"], m["away"]
        ph, pa = float(m["p_home_win"]), float(m["p_away_win"])
        fav = home if ph >= pa else away
        print(f"  {home:18s} {ph:5.1%}  vs  {away:18s} {pa:5.1%}  → fav: {fav}")

    br = data.get("most_likely_bracket", {})
    mwp = data.get("match_win_probs", [])
    frac = data.get("most_likely_bracket_fraction", 0)
    count = data.get("most_likely_bracket_count", 0)
    print(
        f"\n--- Most common complete path ({frac:.2%}, n={count}) — illustrative only ---"
    )
    for rnd in _path_rounds(data, history):
        print(f"\n  {ROUND_LABELS[rnd]}")
        for m in br.get(rnd, []):
            ph, pa = _match_prob(mwp, m["home"], m["away"], rnd)
            prob_str = ""
            if ph is not None:
                prob_str = f"  [{m['home']} {ph:.0%} | {m['away']} {pa:.0%}]"
            print(
                f"    {m['home']} vs {m['away']}  {m['score']:>5s}  "
                f"→ {m['winner']}{prob_str}"
            )
    final = br.get("final", {})
    if final:
        print(f"\n  {ROUND_LABELS['final']}")
        print(
            f"    {final['home']} vs {final['away']}  {final['score']}  "
            f"→ {final['winner']}"
        )
    print("=" * 72)


def _r16_match_html(home: str, away: str, p_home: float, p_away: float) -> str:
    def cell(name: str, p: float, is_fav: bool) -> str:
        cls = "team favorite" if is_fav else "team"
        return (
            f'<div class="{cls}">{html.escape(name)}'
            f'<span class="pct">{p:.0%} chance to win</span></div>'
        )

    fav_home = p_home >= p_away
    return f"""
    <div class="match">
      {cell(home, p_home, fav_home)}
      <div class="vs">vs</div>
      {cell(away, p_away, not fav_home)}
    </div>"""


def _pending_match_html(home: str, away: str) -> str:
    return f"""
    <div class="match">
      <div class="team">{html.escape(home)}</div>
      <div class="vs">vs</div>
      <div class="team">{html.escape(away)}</div>
    </div>"""


def _scored_match_html(
    home: str,
    away: str,
    p_home: float,
    p_away: float,
    *,
    correct: bool,
    actual_winner: str,
    score: str,
) -> str:
    def cell(name: str, p: float, is_fav: bool) -> str:
        cls = "team favorite" if is_fav else "team"
        return (
            f'<div class="{cls}">{html.escape(name)}'
            f'<span class="pct">{p:.0%} chance to win</span></div>'
        )

    fav_home = p_home >= p_away
    row_cls = "pick-correct" if correct else "pick-wrong"
    return f"""
    <div class="match scored {row_cls}">
      {cell(home, p_home, fav_home)}
      <div class="vs">vs<div class="actual">{html.escape(score)} → {html.escape(actual_winner)}</div></div>
      {cell(away, p_away, not fav_home)}
    </div>"""


def _path_match_html(
    home: str,
    away: str,
    winner: str,
    score: str,
    p_home: float | None,
    p_away: float | None,
) -> str:
    def cell(name: str, p: float | None) -> str:
        cls = "team path-winner" if name == winner else "team"
        pct = f'<span class="pct">{p:.1%}</span>' if p is not None else ""
        return f'<div class="{cls}">{html.escape(name)}{pct}</div>'

    return f"""
    <div class="match path-match">
      {cell(home, p_home)}
      <div class="score">{html.escape(score)}</div>
      {cell(away, p_away)}
    </div>"""


def write_html(data: dict, out_path: Path, history: dict | None = None) -> None:
    n_sims = data.get("n_sims", 0)
    champ_probs = data.get("champion_probs", {})
    adv = data.get("advancement_probs", {})
    mwp = data.get("match_win_probs", [])
    br = data.get("most_likely_bracket", {})
    frac = float(data.get("most_likely_bracket_fraction", 0))
    count = int(data.get("most_likely_bracket_count", 0))
    adv_cols = _advancement_columns(data, history)
    path_rounds = _path_rounds(data, history)

    top_teams = sorted(champ_probs.items(), key=lambda x: -x[1])[:10]

    outlook_header = "".join(f"<th>{label}</th>" for _, label in adv_cols)
    outlook_rows = ""
    for team, _ in top_teams:
        row = adv.get(team, {})
        cells = []
        for i, (key, _) in enumerate(adv_cols):
            val = row.get(key, 0)
            if i == 0:
                cells.append(
                    f'<td class="num win-cell"><div class="win-cell-wrap">'
                    f'<span class="win-pct">{val:.1%}</span>'
                    f'<div class="bar-track"><div class="bar" style="width:{min(val * 100, 100):.1f}%"></div></div>'
                    f"</div></td>"
                )
            else:
                cells.append(f'<td class="num">{val:.1%}</td>')
        outlook_rows += f"<tr><td>{html.escape(team)}</td>{''.join(cells)}</tr>"

    history_sections = ""
    if history:
        for rnd_entry in reversed(history.get("rounds", [])):
            summary = rnd_entry.get("summary", {})
            label = rnd_entry.get("label", ROUND_LABELS.get(rnd_entry.get("round", ""), "Results"))
            hist_sims = rnd_entry.get("n_sims", 0)
            correct = summary.get("correct", 0)
            total = summary.get("total", 0)
            rows_html = "".join(
                _scored_match_html(
                    m["home"],
                    m["away"],
                    float(m["p_home_win"]),
                    float(m["p_away_win"]),
                    correct=bool(m.get("correct")),
                    actual_winner=m["winner"],
                    score=m.get("score", ""),
                )
                for m in rnd_entry.get("results", [])
            )
            history_sections += f"""
  <h2>{html.escape(label)} predictions</h2>
  <p class="section-note">From {hist_sims:,} simulations before the round was played. {correct}/{total} correct.</p>
  <section>{rows_html}</section>"""

    active_round = _active_round(data, history)
    active_label = ROUND_LABELS[active_round]
    match_html = "".join(
        _r16_match_html(
            m["home"],
            m["away"],
            float(m["p_home_win"]),
            float(m["p_away_win"]),
        )
        for m in _round_matches(mwp, active_round)
    )
    active_matches_html = f"""
  <h2>{active_label} — who wins?</h2>
  <p class="section-note">Win chance for each team. Blue = more likely to win.</p>
  <section>{match_html}</section>"""
    if active_round == "final":
        bronze_html = "".join(
            _r16_match_html(
                m["home"],
                m["away"],
                float(m["p_home_win"]),
                float(m["p_away_win"]),
            )
            for m in _round_matches(mwp, "third_place")
        )
        active_matches_html += f"""
  <h2>Third-place match — who wins?</h2>
  <p class="section-note">Win chance for each team. Blue = more likely to win.</p>
  <section>{bronze_html}</section>"""

    path_sections: list[str] = []
    for rnd in path_rounds:
        matches = []
        for m in br.get(rnd, []):
            ph, pa = _match_prob(mwp, m["home"], m["away"], rnd)
            matches.append(
                _path_match_html(m["home"], m["away"], m["winner"], m["score"], ph, pa)
            )
        if matches:
            path_sections.append(
                f'<h3 class="path-round">{ROUND_LABELS[rnd]}</h3>{"".join(matches)}'
            )
    final = br.get("final", {})
    if final:
        ph, pa = _match_prob(mwp, final["home"], final["away"], "final")
        path_sections.append(
            f'<h3 class="path-round">{ROUND_LABELS["final"]}</h3>'
            f'{_path_match_html(final["home"], final["away"], final["winner"], final["score"], ph, pa)}'
        )
    third_place = br.get("third_place", {})
    if third_place:
        ph, pa = _match_prob(
            mwp, third_place["home"], third_place["away"], "third_place"
        )
        path_sections.insert(
            0,
            f'<h3 class="path-round">{ROUND_LABELS["third_place"]}</h3>'
            f'{_path_match_html(third_place["home"], third_place["away"], third_place["winner"], third_place["score"], ph, pa)}',
        )

    path_champion = html.escape(str(br.get("champion", "?")))
    path_pct = f"{frac:.2%}"
    path_note = (
        f"This is the knockout tree that showed up most often — "
        f"{count:,} of {n_sims:,} runs ({path_pct}). "
        f"Most simulations still play out differently; the table above is the main forecast."
    )
    if data.get("completed"):
        champion = html.escape(
            str(data.get("champion") or (history or {}).get("champion", "?"))
        )
        forecast_sections = f"""
  <h2>Champion</h2>
  <p class="section-note">Spain won the 2026 World Cup, 1–0 after extra time against Argentina.</p>
  <section class="path-section">
    <div class="path-champion">World champions: <strong>{champion}</strong></div>
  </section>"""
    elif data.get("pending"):
        fixtures = data.get("fixtures", {})
        final_fixture = fixtures.get("final", [])
        bronze_fixture = fixtures.get("third_place", [])
        forecast_sections = f"""
  <h2>Final forecast pending</h2>
  <p class="section-note">The 80,000-simulation final forecast is being prepared.</p>
  <h2>Final</h2>
  <section>{_pending_match_html(*final_fixture)}</section>
  <h2>Third-place match</h2>
  <section>{_pending_match_html(*bronze_fixture)}</section>"""
    else:
        forecast_sections = f"""
  <h2>Who goes how far?</h2>
  <p class="section-note">
    How often each team wins the cup or reaches each round. Numbers are percentages.
  </p>
  <table class="outlook-table">
    <thead><tr><th>Team</th>{outlook_header}</tr></thead>
    <tbody>{outlook_rows}</tbody>
  </table>

{active_matches_html}

  <h2>{"Most likely results" if active_round == "final" else "Most likely path"}</h2>
  <p class="section-note">{path_note}</p>
  <section class="path-section">
    {"".join(path_sections)}
    <div class="path-champion">Champion on this path: <strong>{path_champion}</strong></div>
  </section>"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>WC 2026 Forecast</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      max-width: 760px;
      margin: 2rem auto;
      padding: 0 1rem 3rem;
      background: #0f1419;
      color: #e7e9ea;
      line-height: 1.45;
    }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
    h2 {{
      font-size: 0.95rem;
      color: #8899a6;
      margin: 2rem 0 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    h3.path-round {{
      font-size: 0.85rem;
      color: #8899a6;
      margin: 1rem 0 0.35rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .meta {{ color: #8899a6; font-size: 0.9rem; margin-bottom: 0.5rem; }}
    .section-note {{
      color: #8899a6;
      font-size: 0.88rem;
      margin: -0.35rem 0 0.75rem;
      line-height: 1.45;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 0.4rem 0.5rem; text-align: left; vertical-align: middle; }}
    th {{
      color: #8899a6;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-bottom: 1px solid #2f3336;
    }}
    td.num, th {{ text-align: right; }}
    td:first-child, th:first-child {{ text-align: left; }}
    td.win-cell {{ vertical-align: middle; }}
    .win-cell-wrap {{
      display: inline-flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.3rem;
      min-width: 4.25rem;
    }}
    .win-pct {{ font-weight: 600; color: #e7e9ea; line-height: 1.2; }}
    .bar-track {{
      width: 4.25rem;
      height: 5px;
      background: #2f3336;
      border-radius: 3px;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      background: #1d9bf0;
      border-radius: 3px;
      min-width: 2px;
    }}
    tr:not(:last-child) td {{ border-bottom: 1px solid #2f3336; }}
    .match {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 0.5rem;
      align-items: center;
      padding: 0.55rem 0;
      border-bottom: 1px solid #2f3336;
    }}
    .match.scored {{
      border-radius: 6px;
      padding: 0.55rem 0.35rem;
      margin-bottom: 0.25rem;
      border: 1px solid transparent;
    }}
    .match.pick-correct {{
      background: rgba(0, 186, 124, 0.08);
      border-color: rgba(0, 186, 124, 0.35);
    }}
    .match.pick-wrong {{
      background: rgba(244, 33, 46, 0.08);
      border-color: rgba(244, 33, 46, 0.35);
    }}
    .team {{ padding: 0.35rem 0.5rem; border-radius: 4px; }}
    .team.favorite {{ background: #1a2a3a; font-weight: 600; color: #1d9bf0; }}
    .team.path-winner {{ background: #1d3a2a; font-weight: 600; color: #00ba7c; }}
    .pct {{
      display: block;
      font-size: 0.75rem;
      color: #8899a6;
      font-weight: normal;
      margin-top: 0.15rem;
    }}
    .vs {{ color: #8899a6; font-size: 0.85rem; text-align: center; min-width: 2rem; }}
    .vs .actual {{
      font-size: 0.72rem;
      color: #8899a6;
      margin-top: 0.2rem;
      white-space: nowrap;
    }}
    .score {{ font-weight: 700; color: #fff; text-align: center; min-width: 3rem; }}
    .path-section {{
      margin-top: 0.5rem;
      padding: 1rem;
      background: #1a1f26;
      border-radius: 8px;
      border: 1px solid #2f3336;
    }}
    .path-champion {{
      text-align: center;
      padding: 0.75rem;
      margin-top: 0.5rem;
      background: #0f1419;
      border-radius: 6px;
      color: #8899a6;
      font-size: 0.9rem;
    }}
    .footer {{
      margin-top: 2rem;
      padding-top: 1rem;
      border-top: 1px solid #2f3336;
      font-size: 0.82rem;
      color: #8899a6;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <h1>WC 2026 Knockout Forecast</h1>
  <p class="meta">{_meta_line(data, history)}</p>

{forecast_sections}
{history_sections}

  <p class="footer">
    The model simulates every knockout match thousands of times, picks a scoreline each time,
    and tracks who advances. These are averages over all those runs, not one predicted bracket.
  </p>
</body>
</html>"""
    out_path.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Display WC 2026 forecast results")
    parser.add_argument(
        "forecast",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "output" / "wc2026_forecast.json",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Round history JSON (completed rounds + scoring)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write HTML report (default: <forecast>.html)",
    )
    parser.add_argument("--no-print", action="store_true")
    args = parser.parse_args()

    if not args.forecast.exists():
        print(f"ERROR: Missing {args.forecast}", file=sys.stderr)
        return 1

    data = _load(args.forecast)
    history = _load(args.history) if args.history and args.history.exists() else None

    if not args.no_print:
        print_report(data, history)

    html_path = args.html
    if html_path is None:
        html_path = args.forecast.with_suffix(".html")
    write_html(data, html_path, history)
    print(f"\nHTML report: {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
