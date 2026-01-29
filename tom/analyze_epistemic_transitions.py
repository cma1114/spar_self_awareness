#!/usr/bin/env python3
"""
Analyze epistemic category transitions (ECTs) and situation tracking per scenario.

Uses the category-based ECT model (see EPISTEMIC_METRICS.md):
  - Certainty axis: knowledge ↔ belief (#1 and #2)
  - Accuracy axis: true ↔ false belief (#3 and #4)

Also reports situation tracking (visible events from A's perspective) to
verify that Extra=0 and Extra=1 have been equalized by filler events.
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
            extra = sc.extra if sc.extra else 0
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

    # === TABLE 1: ECT comparison ===
    lines.append("TABLE 1: ECT COMPARISON (Extra=0 vs Extra=1)")
    lines.append("-" * 90)
    header1 = (f"{'ID':>4s}  {'E0 ECT':>8s}  {'E1 ECT':>8s}  {'Delta':>8s}  "
               f"{'E0 cert':>8s}  {'E0 acc':>8s}  {'E1 cert':>8s}  {'E1 acc':>8s}  {'Flag':>6s}")
    lines.append(header1)
    lines.append("-" * len(header1))

    ect_flagged = []
    for sid in unique_ids:
        has_extra1 = any(s['Extra'] == 1 for s in spec_by_id[sid])
        if not has_extra1:
            e0_total = [ect_results[sid].get((0, s), {}).get('total', None) for s in seeds]
            e0_total = [t for t in e0_total if t is not None]
            lines.append(f"{sid:>4s}  {range_str(e0_total):>8s}  {'--':>8s}  {'--':>8s}  "
                         f"{'--':>8s}  {'--':>8s}  {'--':>8s}  {'--':>8s}  {'':>6s}")
            continue

        deltas = []
        e0_totals, e1_totals = [], []
        e0_certs, e0_accs, e1_certs, e1_accs = [], [], [], []
        for s in seeds:
            t0 = ect_results[sid].get((0, s))
            t1 = ect_results[sid].get((1, s))
            if t0 is not None and t1 is not None:
                deltas.append(t1['total'] - t0['total'])
                e0_totals.append(t0['total'])
                e1_totals.append(t1['total'])
                e0_certs.append(t0['certainty'])
                e0_accs.append(t0['accuracy'])
                e1_certs.append(t1['certainty'])
                e1_accs.append(t1['accuracy'])

        if not deltas:
            lines.append(f"{sid:>4s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  "
                         f"{'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'':>6s}")
            continue

        flag = ""
        if min(deltas) <= 0:
            flag = "FAIL"
            ect_flagged.append((sid, min(deltas), max(deltas)))

        lines.append(f"{sid:>4s}  {range_str(e0_totals):>8s}  {range_str(e1_totals):>8s}  "
                     f"{range_str(deltas):>8s}  {range_str(e0_certs):>8s}  {range_str(e0_accs):>8s}  "
                     f"{range_str(e1_certs):>8s}  {range_str(e1_accs):>8s}  {flag:>6s}")

    # === TABLE 2: Situation tracking comparison ===
    lines.append("")
    lines.append("")
    lines.append("TABLE 2: SITUATION TRACKING COMPARISON (Extra=0 vs Extra=1)")
    lines.append("-" * 70)
    header2 = (f"{'ID':>4s}  {'E0 SIT':>8s}  {'E1 SIT':>8s}  {'Delta':>8s}  "
               f"{'E0 RAW':>8s}  {'E1 RAW':>8s}  {'Flag':>6s}")
    lines.append(header2)
    lines.append("-" * len(header2))

    sit_flagged = []
    for sid in unique_ids:
        has_extra1 = any(s['Extra'] == 1 for s in spec_by_id[sid])
        if not has_extra1:
            e0_sit = [sit_results[sid].get((0, s), None) for s in seeds]
            e0_sit = [v for v in e0_sit if v is not None]
            e0_raw = [raw_event_counts[sid].get((0, s), None) for s in seeds]
            e0_raw = [v for v in e0_raw if v is not None]
            lines.append(f"{sid:>4s}  {range_str(e0_sit):>8s}  {'--':>8s}  {'--':>8s}  "
                         f"{range_str(e0_raw):>8s}  {'--':>8s}  {'':>6s}")
            continue

        deltas = []
        e0_sits, e1_sits = [], []
        e0_raws, e1_raws = [], []
        for s in seeds:
            s0 = sit_results[sid].get((0, s))
            s1 = sit_results[sid].get((1, s))
            r0 = raw_event_counts[sid].get((0, s))
            r1 = raw_event_counts[sid].get((1, s))
            if s0 is not None and s1 is not None:
                deltas.append(s1 - s0)
                e0_sits.append(s0)
                e1_sits.append(s1)
            if r0 is not None:
                e0_raws.append(r0)
            if r1 is not None:
                e1_raws.append(r1)

        if not deltas:
            lines.append(f"{sid:>4s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  "
                         f"{'N/A':>8s}  {'N/A':>8s}  {'':>6s}")
            continue

        flag = ""
        # Flag if the gap is too large (> 3 in either direction for any seed)
        if max(abs(d) for d in deltas) > 3:
            flag = "GAP"
            sit_flagged.append((sid, min(deltas), max(deltas)))

        lines.append(f"{sid:>4s}  {range_str(e0_sits):>8s}  {range_str(e1_sits):>8s}  "
                     f"{range_str(deltas):>8s}  {range_str(e0_raws):>8s}  {range_str(e1_raws):>8s}  "
                     f"{flag:>6s}")

    # === SUMMARY ===
    lines.append("")
    lines.append("")
    lines.append("=" * 90)
    lines.append("SUMMARY")
    lines.append("")

    lines.append("--- ECT (Extra=1 should always have more ECTs than Extra=0) ---")
    if ect_flagged:
        lines.append(f"FLAGGED: {len(ect_flagged)} scenario(s) where Extra=1 ECT <= Extra=0:")
        for sid, dmin, dmax in ect_flagged:
            lines.append(f"  ID {sid}: ECT delta range [{dmin}, {dmax}]")
    else:
        lines.append("PASS: All scenarios have Extra=1 ECT > Extra=0 for every seed.")
    lines.append("")

    lines.append("--- Situation Tracking (Extra=0 and Extra=1 should be similar) ---")
    if sit_flagged:
        lines.append(f"FLAGGED: {len(sit_flagged)} scenario(s) with visible event gap > 3:")
        for sid, dmin, dmax in sit_flagged:
            lines.append(f"  ID {sid}: SIT delta range [{dmin}, {dmax}]")
    else:
        lines.append("PASS: All scenarios have Extra=0 and Extra=1 visible events within ±3.")
    lines.append("")

    # === FLAGGED ECT DETAIL ===
    if ect_flagged:
        lines.append("")
        lines.append("ECT FLAGGED SCENARIO DETAIL (per-seed)")
        lines.append("=" * 90)
        for sid, dmin, dmax in ect_flagged:
            lines.append("")
            lines.append(f"--- ID {sid} ---")
            seed_header = (f"  {'Seed':>4s}  {'E0 tot':>6s}  {'E1 tot':>6s}  {'Delta':>6s}  "
                           f"{'E0 cer':>6s}  {'E0 acc':>6s}  {'E1 cer':>6s}  {'E1 acc':>6s}  "
                           f"{'E0 SIT':>6s}  {'E1 SIT':>6s}")
            lines.append(seed_header)
            for s in seeds:
                t0 = ect_results[sid].get((0, s))
                t1 = ect_results[sid].get((1, s))
                s0 = sit_results[sid].get((0, s))
                s1 = sit_results[sid].get((1, s))
                if t0 is not None and t1 is not None:
                    delta = t1['total'] - t0['total']
                    flag_mark = " <--" if delta <= 0 else ""
                    lines.append(
                        f"  {s:>4d}  {t0['total']:>6d}  {t1['total']:>6d}  {delta:>6d}  "
                        f"{t0['certainty']:>6d}  {t0['accuracy']:>6d}  "
                        f"{t1['certainty']:>6d}  {t1['accuracy']:>6d}  "
                        f"{s0 or 0:>6d}  {s1 or 0:>6d}{flag_mark}"
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
    lines.append("  Delta   = Extra=1 value - Extra=0 value")
    lines.append("  FAIL    = Extra=1 ECT <= Extra=0 for some seed")
    lines.append("  GAP     = |SIT delta| > 3 for some seed")

    report = "\n".join(lines)

    outpath = os.path.join(os.path.dirname(__file__), 'tom_llm_logs', 'epistemic_transition_analysis.txt')
    with open(outpath, 'w') as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
