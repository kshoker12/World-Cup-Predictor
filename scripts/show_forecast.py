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
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _r16_lookup(match_win_probs: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for row in match_win_probs:
        if row.get("round") != "round_of_16":
            continue
        key = (row["home"], row["away"])
        out[key] = row
        out[(row["away"], row["home"])] = {
            **row,
            "p_home_win": row["p_away_win"],
            "p_away_win": row["p_home_win"],
        }
    return out


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


def _winner_mark(name: str, winner: str) -> str:
    return f"**{name}**" if name == winner else name


def print_report(data: dict) -> None:
    n_sims = data.get("n_sims", 0)
    print("=" * 72)
    print(f"WC 2026 FORECAST — {n_sims:,} simulations")
    print("=" * 72)

    print("\n--- Champion probabilities (top 10) ---")
    champ = data.get("champion_probs", {})
    for team, prob in sorted(champ.items(), key=lambda x: -x[1])[:10]:
        bar = "#" * int(prob * 40)
        print(f"  {team:20s} {prob:6.2%}  {bar}")

    r16_lookup = _r16_lookup(data.get("match_win_probs", []))
    print("\n--- Round of 16 (model win %) ---")
    for m in data.get("most_likely_bracket", {}).get("round_of_16", []):
        home, away = m["home"], m["away"]
        row = r16_lookup.get((home, away))
        if row:
            ph, pa = row["p_home_win"], row["p_away_win"]
            print(
                f"  {home:18s} {ph:5.1%}  vs  {away:18s} {pa:5.1%}  "
                f"→ likely: {m['winner']} ({m['score']})"
            )
        else:
            print(f"  {home} vs {away} → {m['winner']} ({m['score']})")

    br = data.get("most_likely_bracket", {})
    mwp = data.get("match_win_probs", [])
    frac = data.get("most_likely_bracket_fraction", 0)
    count = data.get("most_likely_bracket_count", 0)
    print(f"\n--- Most likely full bracket (path frequency: {frac:.2%}, n={count}) ---")
    for rnd in ("round_of_16", "quarter_finals", "semi_finals"):
        print(f"\n  {ROUND_LABELS[rnd]}")
        for m in br.get(rnd, []):
            ph, pa = _match_prob(mwp, m["home"], m["away"], rnd)
            prob_str = ""
            if ph is not None:
                prob_str = f"  [{m['home']} {ph:.0%} | {m['away']} {pa:.0%}]"
            w_home = "*" if m["winner"] == m["home"] else " "
            w_away = "*" if m["winner"] == m["away"] else " "
            print(
                f"    {w_home}{m['home']:18s}{w_home} vs "
                f"{w_away}{m['away']:18s}{w_away}  {m['score']:>5s}{prob_str}"
            )

    final = br.get("final", {})
    if final:
        ph, pa = _match_prob(mwp, final["home"], final["away"], "final")
        print(f"\n  {ROUND_LABELS['final']}")
        prob_str = f"  [{final['home']} {ph:.0%} | {final['away']} {pa:.0%}]" if ph else ""
        print(
            f"    {final['home']} vs {final['away']}  "
            f"{final['score']}  → {final['winner']}{prob_str}"
        )
    print(f"\n  🏆 Champion (most likely path): {br.get('champion', '?')}")
    print("=" * 72)


def _match_html(
    home: str,
    away: str,
    winner: str,
    score: str,
    p_home: float | None,
    p_away: float | None,
) -> str:
    def cell(name: str, p: float | None) -> str:
        cls = "team winner" if name == winner else "team"
        pct = f'<span class="pct">{p:.1%}</span>' if p is not None else ""
        return f'<div class="{cls}">{html.escape(name)}{pct}</div>'

    return f"""
    <div class="match">
      {cell(home, p_home)}
      <div class="score">{html.escape(score)}</div>
      {cell(away, p_away)}
    </div>"""


def write_html(data: dict, out_path: Path) -> None:
    br = data.get("most_likely_bracket", {})
    mwp = data.get("match_win_probs", [])
    n_sims = data.get("n_sims", 0)
    frac = data.get("most_likely_bracket_fraction", 0)
    champ_probs = sorted(
        data.get("champion_probs", {}).items(), key=lambda x: -x[1]
    )[:10]

    sections: list[str] = []
    for rnd in ("round_of_16", "quarter_finals", "semi_finals"):
        matches_html = []
        for m in br.get(rnd, []):
            ph, pa = _match_prob(mwp, m["home"], m["away"], rnd)
            matches_html.append(
                _match_html(m["home"], m["away"], m["winner"], m["score"], ph, pa)
            )
        sections.append(
            f'<section><h2>{ROUND_LABELS[rnd]}</h2>{"".join(matches_html)}</section>'
        )

    final = br.get("final", {})
    if final:
        ph, pa = _match_prob(mwp, final["home"], final["away"], "final")
        sections.append(
            f'<section><h2>Final</h2>{_match_html(final["home"], final["away"], final["winner"], final["score"], ph, pa)}</section>'
        )

    champ_rows = "".join(
        f'<tr><td>{html.escape(t)}</td><td>{p:.2%}</td>'
        f'<td><div class="bar" style="width:{p*100:.1f}%"></div></td></tr>'
        for t, p in champ_probs
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>WC 2026 Forecast</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; background: #0f1419; color: #e7e9ea; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1rem; color: #8899a6; margin: 1.5rem 0 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .meta {{ color: #8899a6; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .match {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #2f3336; }}
    .team {{ padding: 0.35rem 0.5rem; border-radius: 4px; }}
    .team.winner {{ background: #1d3a2a; font-weight: 600; color: #00ba7c; }}
    .pct {{ display: block; font-size: 0.75rem; color: #8899a6; font-weight: normal; }}
    .score {{ font-weight: 700; color: #fff; text-align: center; min-width: 3rem; }}
    .trophy {{ font-size: 1.25rem; margin-top: 1rem; padding: 1rem; background: #1a1f26; border-radius: 8px; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    td {{ padding: 0.35rem 0.5rem; vertical-align: middle; }}
    .bar {{ height: 8px; background: #1d9bf0; border-radius: 4px; max-width: 200px; }}
  </style>
</head>
<body>
  <h1>WC 2026 Knockout Forecast</h1>
  <p class="meta">{n_sims:,} simulations · most likely path: {frac:.2%} of runs</p>
  {"".join(sections)}
  <div class="trophy">🏆 Most likely path champion: <strong>{html.escape(str(br.get("champion", "?")))}</strong></div>
  <h2>Champion probabilities</h2>
  <table>{champ_rows}</table>
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
    if not args.no_print:
        print_report(data)

    html_path = args.html
    if html_path is None:
        html_path = args.forecast.with_suffix(".html")
    write_html(data, html_path)
    print(f"\nHTML report: {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
