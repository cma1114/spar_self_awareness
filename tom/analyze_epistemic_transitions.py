#!/usr/bin/env python3
"""
Analyze epistemic category transitions (ECTs) and situation tracking per scenario.

Uses the category-based ECT model (see EPISTEMIC_METRICS.md):
  - Certainty axis: knowledge ↔ belief (#1 and #2)
  - Accuracy axis: true ↔ false belief (#3 and #4)

Also reports situation tracking (visible events from A's perspective) to
verify that Extra=1A and Extra=1B have been equalized by filler events.

Extra values:
  - '0A': Minimal events, no ECTs (new)
  - '0B': Higher event load, no ECTs (new)
  - '1A': Filler for SIT parity (was Extra=0)
  - '1B': Extra events with ECT addition (was Extra=1)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tom_helpers import read_specs_from_csv, Scenario, Event
from generate_tom_scenarios_new import (
    generate_scenarios_from_tuples, count_epistemic_category_transitions,
    _count_visible_events, CharacterType, CONTAINERS_GEN,
)

import json
import tempfile
from collections import defaultdict
from typing import Dict, Tuple


def normalize_extra(val):
    """Normalize Extra field to string format for backward compatibility.

    Legacy int values are converted to new string format:
    - None or 0 → '1A' (legacy Extra=0 behavior)
    - 1 → '1B' (legacy Extra=1 behavior)
    """
    if val is None or val == 0: return '1A'  # Legacy Extra=0 → 1A
    if val == 1: return '1B'                  # Legacy Extra=1 → 1B
    if val in ('0A', '0B', '1A', '1B'): return val
    return str(val)


def main():
    specs = read_specs_from_csv(os.path.join(os.path.dirname(__file__), 'ToM - scenarios.csv'))

    # Build lookup: id -> list of specs (extra=0 and extra=1)
    spec_by_id: Dict[str, list] = defaultdict(list)
    for s in specs:
        spec_by_id[s['Id']].append(s)

    # Get unique scenario IDs preserving order
    seen = set()
    unique_ids = []
    for s in specs:
        if s['Id'] not in seen:
            seen.add(s['Id'])
            unique_ids.append(s['Id'])

    seeds = list(range(1, 9))
    chartypes = [
        CharacterType.LIVE_PLAYER,
        CharacterType.DISHONEST_OPPONENT,
        CharacterType.DISHONEST_TEAMMATE,
        CharacterType.DISHONEST_OPPONENT,
    ]

    # Per-scenario, per-(extra, seed) metrics
    ect_results: Dict[str, Dict[Tuple[int, int], Dict[str, int]]] = defaultdict(dict)
    sit_results: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)
    raw_event_counts: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(dict)

    for seed in seeds:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmpfile = f.name
        try:
            generate_scenarios_from_tuples(specs, tmpfile, seed=seed, chartypes=chartypes)
            with open(tmpfile) as f:
                data = json.load(f)
            scenarios = [Scenario.from_dict(s) for s in data['scenarios']]
        finally:
            os.unlink(tmpfile)

        for sc in scenarios:
            sid = sc.id
            extra = normalize_extra(sc.extra)
            ect = count_epistemic_category_transitions(sc)
            vis = _count_visible_events(sc)
            ect_results[sid][(extra, seed)] = ect
            sit_results[sid][(extra, seed)] = vis
            raw_event_counts[sid][(extra, seed)] = len(sc.events)

    # --- Build report ---
    def range_str(vals):
        if not vals:
            return "N/A"
        if min(vals) == max(vals):
            return str(vals[0])
        return f"{min(vals)}-{max(vals)}"

    lines = []
    lines.append("EPISTEMIC CATEGORY TRANSITION (ECT) & SITUATION TRACKING ANALYSIS")
    lines.append("=" * 90)
    lines.append("")
    lines.append("ECT = epistemic category transitions (certainty + accuracy axes)")
    lines.append("SIT = situation tracking events (visible to player A)")
    lines.append("RAW = total raw event count (including invisible)")
    lines.append(f"Seeds: {seeds}")
    lines.append(f"Scenarios: {len(unique_ids)} unique IDs")
    lines.append("")

    # === TABLE 1: ECT comparison (1A vs 1B) ===
    lines.append("TABLE 1: ECT COMPARISON (Extra=1A vs Extra=1B)")
    lines.append("-" * 90)
    header1 = (f"{'ID':>4s}  {'1A ECT':>8s}  {'1B ECT':>8s}  {'Delta':>8s}  "
               f"{'1A cert':>8s}  {'1A acc':>8s}  {'1B cert':>8s}  {'1B acc':>8s}  {'Flag':>6s}")
    lines.append(header1)
    lines.append("-" * len(header1))

    ect_flagged = []
    for sid in unique_ids:
        has_extra1b = any(s['Extra'] == '1B' for s in spec_by_id[sid])
        if not has_extra1b:
            e1a_total = [ect_results[sid].get(('1A', s), {}).get('total', None) for s in seeds]
            e1a_total = [t for t in e1a_total if t is not None]
            lines.append(f"{sid:>4s}  {range_str(e1a_total):>8s}  {'--':>8s}  {'--':>8s}  "
                         f"{'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'':>6s}")
            continue

        deltas = []
        e1a_totals, e1b_totals = [], []
        e1a_certs, e1a_accs, e1b_certs, e1b_accs = [], [], [], []
        for s in seeds:
            t1a = ect_results[sid].get(('1A', s))
            t1b = ect_results[sid].get(('1B', s))
            if t1a is not None and t1b is not None:
                deltas.append(t1b['total'] - t1a['total'])
                e1a_totals.append(t1a['total'])
                e1b_totals.append(t1b['total'])
                e1a_certs.append(t1a['certainty'])
                e1a_accs.append(t1a['accuracy'])
                e1b_certs.append(t1b['certainty'])
                e1b_accs.append(t1b['accuracy'])

        if not deltas:
            lines.append(f"{sid:>4s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  "
                         f"{'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'':>6s}")
            continue

        flag = ""
        if min(deltas) <= 0:
            flag = "FAIL"
            ect_flagged.append((sid, min(deltas), max(deltas)))

        lines.append(f"{sid:>4s}  {range_str(e1a_totals):>8s}  {range_str(e1b_totals):>8s}  "
                     f"{range_str(deltas):>8s}  {range_str(e1a_certs):>8s}  {range_str(e1a_accs):>8s}  "
                     f"{range_str(e1b_certs):>8s}  {range_str(e1b_accs):>8s}  {flag:>6s}")

    # === TABLE 2: Situation tracking comparison (1A vs 1B) ===
    lines.append("")
    lines.append("")
    lines.append("TABLE 2: SITUATION TRACKING COMPARISON (Extra=1A vs Extra=1B)")
    lines.append("-" * 70)
    header2 = (f"{'ID':>4s}  {'1A SIT':>8s}  {'1B SIT':>8s}  {'Delta':>8s}  "
               f"{'1A RAW':>8s}  {'1B RAW':>8s}  {'Flag':>6s}")
    lines.append(header2)
    lines.append("-" * len(header2))

    sit_flagged = []
    for sid in unique_ids:
        has_extra1b = any(s['Extra'] == '1B' for s in spec_by_id[sid])
        if not has_extra1b:
            e1a_sit = [sit_results[sid].get(('1A', s), None) for s in seeds]
            e1a_sit = [v for v in e1a_sit if v is not None]
            e1a_raw = [raw_event_counts[sid].get(('1A', s), None) for s in seeds]
            e1a_raw = [v for v in e1a_raw if v is not None]
            lines.append(f"{sid:>4s}  {range_str(e1a_sit):>8s}  {'--':>8s}  {'--':>8s}  "
                         f"{range_str(e1a_raw):>8s}  {'--':>8s}  {'':>6s}")
            continue

        deltas = []
        e1a_sits, e1b_sits = [], []
        e1a_raws, e1b_raws = [], []
        for s in seeds:
            s1a = sit_results[sid].get(('1A', s))
            s1b = sit_results[sid].get(('1B', s))
            r1a = raw_event_counts[sid].get(('1A', s))
            r1b = raw_event_counts[sid].get(('1B', s))
            if s1a is not None and s1b is not None:
                deltas.append(s1b - s1a)
                e1a_sits.append(s1a)
                e1b_sits.append(s1b)
            if r1a is not None:
                e1a_raws.append(r1a)
            if r1b is not None:
                e1b_raws.append(r1b)

        if not deltas:
            lines.append(f"{sid:>4s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  "
                         f"{'N/A':>8s}  {'N/A':>8s}  {'':>6s}")
            continue

        flag = ""
        # Flag if the gap is too large (> 3 in either direction for any seed)
        if max(abs(d) for d in deltas) > 3:
            flag = "GAP"
            sit_flagged.append((sid, min(deltas), max(deltas)))

        lines.append(f"{sid:>4s}  {range_str(e1a_sits):>8s}  {range_str(e1b_sits):>8s}  "
                     f"{range_str(deltas):>8s}  {range_str(e1a_raws):>8s}  {range_str(e1b_raws):>8s}  "
                     f"{flag:>6s}")

    # === SUMMARY ===
    lines.append("")
    lines.append("")
    lines.append("=" * 90)
    lines.append("SUMMARY")
    lines.append("")

    lines.append("--- ECT (Extra=1B should always have more ECTs than Extra=1A) ---")
    if ect_flagged:
        lines.append(f"FLAGGED: {len(ect_flagged)} scenario(s) where Extra=1B ECT <= Extra=1A:")
        for sid, dmin, dmax in ect_flagged:
            lines.append(f"  ID {sid}: ECT delta range [{dmin}, {dmax}]")
    else:
        lines.append("PASS: All scenarios have Extra=1B ECT > Extra=1A for every seed.")
    lines.append("")

    lines.append("--- Situation Tracking (Extra=1A and Extra=1B should be similar) ---")
    if sit_flagged:
        lines.append(f"FLAGGED: {len(sit_flagged)} scenario(s) with visible event gap > 3:")
        for sid, dmin, dmax in sit_flagged:
            lines.append(f"  ID {sid}: SIT delta range [{dmin}, {dmax}]")
    else:
        lines.append("PASS: All scenarios have Extra=1A and Extra=1B visible events within ±3.")
    lines.append("")

    # === FLAGGED ECT DETAIL ===
    if ect_flagged:
        lines.append("")
        lines.append("ECT FLAGGED SCENARIO DETAIL (per-seed)")
        lines.append("=" * 90)
        for sid, dmin, dmax in ect_flagged:
            lines.append("")
            lines.append(f"--- ID {sid} ---")
            seed_header = (f"  {'Seed':>4s}  {'1A tot':>6s}  {'1B tot':>6s}  {'Delta':>6s}  "
                           f"{'1A cer':>6s}  {'1A acc':>6s}  {'1B cer':>6s}  {'1B acc':>6s}  "
                           f"{'1A SIT':>6s}  {'1B SIT':>6s}")
            lines.append(seed_header)
            for s in seeds:
                t1a = ect_results[sid].get(('1A', s))
                t1b = ect_results[sid].get(('1B', s))
                s1a = sit_results[sid].get(('1A', s))
                s1b = sit_results[sid].get(('1B', s))
                if t1a is not None and t1b is not None:
                    delta = t1b['total'] - t1a['total']
                    flag_mark = " <--" if delta <= 0 else ""
                    lines.append(
                        f"  {s:>4d}  {t1a['total']:>6d}  {t1b['total']:>6d}  {delta:>6d}  "
                        f"{t1a['certainty']:>6d}  {t1a['accuracy']:>6d}  "
                        f"{t1b['certainty']:>6d}  {t1b['accuracy']:>6d}  "
                        f"{s1a or 0:>6d}  {s1b or 0:>6d}{flag_mark}"
                    )

    # === Column legend ===
    lines.append("")
    lines.append("")
    lines.append("COLUMN LEGEND")
    lines.append("-" * 50)
    lines.append("  ECT     = epistemic category transitions (total)")
    lines.append("  cert    = certainty transitions (#1 + #2)")
    lines.append("  acc     = accuracy transitions (#3 + #4)")
    lines.append("  SIT     = visible events from A's perspective")
    lines.append("  RAW     = total raw event count")
    lines.append("  Delta   = Extra=1B value - Extra=1A value")
    lines.append("  FAIL    = Extra=1B ECT <= Extra=1A for some seed")
    lines.append("  GAP     = |SIT delta| > 3 for some seed")
    lines.append("")
    lines.append("Extra field meanings:")
    lines.append("  0A = minimal events, no ECTs (new)")
    lines.append("  0B = higher event load, no ECTs (new)")
    lines.append("  1A = filler for SIT parity (was Extra=0)")
    lines.append("  1B = extra events with ECT addition (was Extra=1)")

    report = "\n".join(lines)

    outpath = os.path.join(os.path.dirname(__file__), 'tom_llm_logs', 'epistemic_transition_analysis.txt')
    with open(outpath, 'w') as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
