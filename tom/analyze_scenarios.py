#!/usr/bin/env python3
"""
Analyze scenario descriptions from ToM test runs.

Loads the N most recent game_data.json files and produces both qualitative
and quantitative analysis of generated scenario complexity.
"""

import json
import glob
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def find_recent_game_data(logs_dir: str, n: int = 8) -> List[str]:
    """Find the N most recent game_data.json files by modification time."""
    pattern = os.path.join(logs_dir, '*_game_data.json')
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:n]


def parse_events_from_desc(desc: str) -> dict:
    """Parse event information from a scenario description string.

    Returns dict with:
      - event_count: total number of events
      - event_types: dict of {type: count}
      - word_count: total words in description
      - char_count: total characters in description
    """
    # Each event is followed by " ..." — count occurrences.
    # Using " ..." (not " ... ") to catch the final marker at end-of-string
    # where there's no trailing space.
    event_count = desc.count(" ...")

    # Split on " ..." to get segments (first segment has intro + first event)
    segments = desc.split(" ...")
    event_types = defaultdict(int)

    for seg in segments:
        seg_lower = seg.lower().strip()
        if not seg_lower:
            continue
        # Classify event type by verb
        if ' puts ' in seg_lower or ' put ' in seg_lower:
            event_types['put'] += 1
        if ' leaves ' in seg_lower or ' leave ' in seg_lower:
            event_types['leave'] += 1
        if ' moves ' in seg_lower or ' move ' in seg_lower:
            event_types['move'] += 1
        if ' enters ' in seg_lower or ' enter ' in seg_lower:
            event_types['enter'] += 1
        if ' removes ' in seg_lower or ' remove ' in seg_lower:
            event_types['remove'] += 1

    return {
        'event_count': event_count,
        'event_types': dict(event_types),
        'word_count': len(desc.split()),
        'char_count': len(desc),
    }


def load_and_parse_all(file_paths: List[str]) -> List[dict]:
    """Load game_data.json files and parse scenario descriptions.

    Returns list of dicts, each with parsed scenario data plus metadata.
    """
    all_parsed = []
    for fpath in file_paths:
        basename = os.path.basename(fpath)
        with open(fpath, 'r') as f:
            records = json.load(f)

        # Filter to player A records only
        a_records = [r for r in records if r.get('character') == 'A']

        for rec in a_records:
            parsed = parse_events_from_desc(rec.get('scenario_desc', ''))
            extra_val = rec.get('extra')
            parsed.update({
                'scenario_id': rec.get('scenario_id', ''),
                'extra': 0 if extra_val is None else int(extra_val),
                'ks_self': rec.get('ks_self', ''),
                'ks_teammate': rec.get('ks_teammate', ''),
                'ks_opponent': rec.get('ks_opponent', ''),
                'correct_action': rec.get('optimal_action', ''),
                'scenario_desc': rec.get('scenario_desc', ''),
                'source_file': basename,
            })
            all_parsed.append(parsed)

    return all_parsed


def compute_per_scenario_stats(parsed_data: List[dict]) -> Dict[Tuple[str, int], dict]:
    """Compute per (scenario_id, extra) statistics across runs.

    Returns dict keyed by (scenario_id, extra) with stats.
    """
    groups = defaultdict(list)
    for p in parsed_data:
        key = (p['scenario_id'], p['extra'])
        groups[key].append(p)

    stats = {}
    for key, items in groups.items():
        event_counts = [it['event_count'] for it in items]
        word_counts = [it['word_count'] for it in items]
        n = len(event_counts)

        stats[key] = {
            'n_runs': n,
            'event_mean': statistics.mean(event_counts),
            'event_stdev': statistics.stdev(event_counts) if n > 1 else 0.0,
            'event_min': min(event_counts),
            'event_max': max(event_counts),
            'word_mean': statistics.mean(word_counts),
            'word_stdev': statistics.stdev(word_counts) if n > 1 else 0.0,
            'ks_self': items[0]['ks_self'],
            'ks_teammate': items[0]['ks_teammate'],
            'ks_opponent': items[0]['ks_opponent'],
            'correct_action': items[0]['correct_action'],
            'event_counts': event_counts,
        }

    return stats


def generate_report(parsed_data: List[dict],
                    stats: Dict[Tuple[str, int], dict],
                    file_paths: List[str]) -> str:
    """Generate the full analysis report."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("SCENARIO DESCRIPTION ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"\nFiles analyzed: {len(file_paths)}")
    for fp in file_paths:
        lines.append(f"  {os.path.basename(fp)}")
    lines.append(f"Total scenario records: {len(parsed_data)}")

    extra0 = [p for p in parsed_data if p['extra'] == 0]
    extra1 = [p for p in parsed_data if p['extra'] == 1]
    lines.append(f"  Extra=0: {len(extra0)}")
    lines.append(f"  Extra=1: {len(extra1)}")

    # =========================================================================
    # 1. Event Count Summary Table
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("1. EVENT COUNT PER SCENARIO (across runs)")
    lines.append("=" * 80)
    lines.append(f"\n{'ID':>4} {'Extra':>5} {'Mean':>6} {'StDev':>6} {'Min':>4} {'Max':>4} {'N':>3} {'Action':>15} {'KS_Self':>15}")
    lines.append("-" * 72)

    sorted_keys = sorted(stats.keys(), key=lambda k: (int(k[0]), k[1]))
    for key in sorted_keys:
        s = stats[key]
        sid, extra = key
        lines.append(
            f"{sid:>4} {extra:>5} {s['event_mean']:>6.1f} {s['event_stdev']:>6.2f} "
            f"{s['event_min']:>4} {s['event_max']:>4} {s['n_runs']:>3} "
            f"{s['correct_action']:>15} {s['ks_self']:>15}"
        )

    # =========================================================================
    # 2. Extra=0 vs Extra=1 Complexity Comparison
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("2. EXTRA=0 vs EXTRA=1 COMPLEXITY")
    lines.append("=" * 80)

    e0_counts = [p['event_count'] for p in extra0]
    e1_counts = [p['event_count'] for p in extra1]
    lines.append(f"\nExtra=0: mean={statistics.mean(e0_counts):.2f}, "
                 f"stdev={statistics.stdev(e0_counts):.2f}, "
                 f"min={min(e0_counts)}, max={max(e0_counts)}")
    lines.append(f"Extra=1: mean={statistics.mean(e1_counts):.2f}, "
                 f"stdev={statistics.stdev(e1_counts):.2f}, "
                 f"min={min(e1_counts)}, max={max(e1_counts)}")
    lines.append(f"Delta (Extra=1 - Extra=0): {statistics.mean(e1_counts) - statistics.mean(e0_counts):.2f} events on average")

    # Per-scenario delta
    lines.append(f"\n{'ID':>4} {'E0 Mean':>8} {'E1 Mean':>8} {'Delta':>7} {'Flag':>6}")
    lines.append("-" * 40)

    failures = []
    for sid in sorted(set(k[0] for k in stats.keys()), key=int):
        e0_key = (sid, 0)
        e1_key = (sid, 1)
        if e0_key in stats and e1_key in stats:
            e0_mean = stats[e0_key]['event_mean']
            e1_mean = stats[e1_key]['event_mean']
            delta = e1_mean - e0_mean
            flag = "WARN" if delta <= 0 else ""
            lines.append(f"{sid:>4} {e0_mean:>8.1f} {e1_mean:>8.1f} {delta:>7.1f} {flag:>6}")
            if delta <= 0:
                failures.append(sid)

    if failures:
        lines.append(f"\nWARNING: {len(failures)} scenario(s) where Extra=1 has ≤ events than Extra=0: {failures}")
    else:
        lines.append("\nAll scenarios have more events in Extra=1 than Extra=0.")

    # Check individual runs for any Extra=1 <= Extra=0
    run_failures = []
    for sid in sorted(set(k[0] for k in stats.keys()), key=int):
        e0_key = (sid, 0)
        e1_key = (sid, 1)
        if e0_key in stats and e1_key in stats:
            e0_counts_list = stats[e0_key]['event_counts']
            e1_counts_list = stats[e1_key]['event_counts']
            for i, (e0c, e1c) in enumerate(zip(e0_counts_list, e1_counts_list)):
                if e1c <= e0c:
                    run_failures.append((sid, i, e0c, e1c))

    if run_failures:
        lines.append(f"\nIndividual run failures (Extra=1 ≤ Extra=0): {len(run_failures)}")
        for sid, run_idx, e0c, e1c in run_failures[:20]:
            lines.append(f"  ID {sid}, run {run_idx}: Extra=0={e0c}, Extra=1={e1c}")
    else:
        lines.append("No individual run failures (Extra=1 always > Extra=0 in every run).")

    # =========================================================================
    # 3. Event Type Distribution
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("3. EVENT TYPE DISTRIBUTION")
    lines.append("=" * 80)

    type_totals = {'ALL': defaultdict(int), 'Extra=0': defaultdict(int), 'Extra=1': defaultdict(int)}
    for p in parsed_data:
        group = 'Extra=0' if p['extra'] == 0 else 'Extra=1'
        for etype, count in p['event_types'].items():
            type_totals['ALL'][etype] += count
            type_totals[group][etype] += count

    all_types = sorted(set().union(*[t.keys() for t in type_totals.values()]))
    lines.append(f"\n{'Type':>10} {'ALL':>8} {'Extra=0':>10} {'Extra=1':>10}")
    lines.append("-" * 42)
    for etype in all_types:
        all_c = type_totals['ALL'].get(etype, 0)
        e0_c = type_totals['Extra=0'].get(etype, 0)
        e1_c = type_totals['Extra=1'].get(etype, 0)
        lines.append(f"{etype:>10} {all_c:>8} {e0_c:>10} {e1_c:>10}")

    total_all = sum(type_totals['ALL'].values())
    total_e0 = sum(type_totals['Extra=0'].values())
    total_e1 = sum(type_totals['Extra=1'].values())
    lines.append(f"{'TOTAL':>10} {total_all:>8} {total_e0:>10} {total_e1:>10}")

    # Proportions
    lines.append(f"\nProportions:")
    lines.append(f"{'Type':>10} {'ALL':>8} {'Extra=0':>10} {'Extra=1':>10}")
    lines.append("-" * 42)
    for etype in all_types:
        all_pct = type_totals['ALL'].get(etype, 0) / total_all * 100 if total_all else 0
        e0_pct = type_totals['Extra=0'].get(etype, 0) / total_e0 * 100 if total_e0 else 0
        e1_pct = type_totals['Extra=1'].get(etype, 0) / total_e1 * 100 if total_e1 else 0
        lines.append(f"{etype:>10} {all_pct:>7.1f}% {e0_pct:>9.1f}% {e1_pct:>9.1f}%")

    # =========================================================================
    # 4. Description Length
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("4. DESCRIPTION LENGTH")
    lines.append("=" * 80)

    for label, subset in [("Extra=0", extra0), ("Extra=1", extra1), ("ALL", parsed_data)]:
        words = [p['word_count'] for p in subset]
        chars = [p['char_count'] for p in subset]
        lines.append(f"\n{label} (N={len(subset)}):")
        lines.append(f"  Words: mean={statistics.mean(words):.1f}, stdev={statistics.stdev(words):.1f}, "
                     f"min={min(words)}, max={max(words)}")
        lines.append(f"  Chars: mean={statistics.mean(chars):.1f}, stdev={statistics.stdev(chars):.1f}, "
                     f"min={min(chars)}, max={max(chars)}")

    # =========================================================================
    # 5. Cross-Run Consistency
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("5. CROSS-RUN CONSISTENCY (Coefficient of Variation)")
    lines.append("=" * 80)
    lines.append("\nNote: seed=None means each run regenerates different events,")
    lines.append("so variance reflects how stable complexity is across regenerations.\n")

    high_cv = []
    cv_values = []
    for key in sorted_keys:
        s = stats[key]
        if s['event_mean'] > 0 and s['n_runs'] > 1:
            cv = s['event_stdev'] / s['event_mean']
            cv_values.append(cv)
            if cv > 0.3:
                high_cv.append((key, cv, s))

    if cv_values:
        lines.append(f"CV across all (id, extra) pairs:")
        lines.append(f"  Mean CV: {statistics.mean(cv_values):.3f}")
        lines.append(f"  Max CV:  {max(cv_values):.3f}")
        lines.append(f"  Pairs with CV > 0.3: {len(high_cv)}/{len(cv_values)}")

    if high_cv:
        lines.append(f"\nHigh-variance scenarios (CV > 0.3):")
        lines.append(f"{'ID':>4} {'Extra':>5} {'Mean':>6} {'StDev':>6} {'CV':>6} {'Counts':>30}")
        lines.append("-" * 62)
        for (sid, extra), cv, s in sorted(high_cv, key=lambda x: x[1], reverse=True):
            counts_str = str(s['event_counts'])
            lines.append(f"{sid:>4} {extra:>5} {s['event_mean']:>6.1f} {s['event_stdev']:>6.2f} {cv:>6.3f} {counts_str:>30}")
    else:
        lines.append("\nNo high-variance scenarios (all CV ≤ 0.3).")

    # =========================================================================
    # 6. Epistemic State Breakdown
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("6. EPISTEMIC STATE BREAKDOWN")
    lines.append("=" * 80)

    for extra_level in [0, 1]:
        lines.append(f"\nExtra={extra_level}:")
        ks_groups = defaultdict(list)
        for p in parsed_data:
            if p['extra'] == extra_level:
                ks_groups[p['ks_self']].append(p['event_count'])

        lines.append(f"  {'KS_Self':>15} {'N':>5} {'Mean':>6} {'StDev':>6} {'Min':>4} {'Max':>4}")
        lines.append("  " + "-" * 46)
        for ks_val in sorted(ks_groups.keys()):
            counts = ks_groups[ks_val]
            n = len(counts)
            mean_val = statistics.mean(counts)
            std_val = statistics.stdev(counts) if n > 1 else 0.0
            lines.append(f"  {ks_val:>15} {n:>5} {mean_val:>6.1f} {std_val:>6.2f} {min(counts):>4} {max(counts):>4}")

    # =========================================================================
    # 7. Qualitative Samples
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("7. QUALITATIVE SAMPLES")
    lines.append("=" * 80)
    lines.append("\nA selection of scenario descriptions for manual review.\n")

    # Use scenarios from the most recent file only
    recent = [p for p in parsed_data if p['source_file'] == os.path.basename(file_paths[0])]

    samples = []

    # Simple Extra=0
    simple = [p for p in recent if p['extra'] == 0 and p['event_count'] <= 3]
    if simple:
        samples.append(("Simple Extra=0", simple[0]))

    # Complex Extra=1
    complex_ = sorted([p for p in recent if p['extra'] == 1], key=lambda x: -x['event_count'])
    if complex_:
        samples.append(("Complex Extra=1 (most events)", complex_[0]))

    # UNKNOWN epistemic
    unknown = [p for p in recent if 'Unknown' in p['ks_self'] or 'Unknown' in p['ks_teammate']]
    if unknown:
        samples.append(("UNKNOWN epistemic state", unknown[0]))

    # Believes X
    believes_x = [p for p in recent if p['ks_self'] == 'Believes X']
    if believes_x:
        samples.append(("Self=Believes X", believes_x[0]))

    # Tell Teammate action (format in game_data is "Tell(B, box, brick)" etc.)
    tell = [p for p in recent if p['correct_action'].startswith('Tell')]
    if tell:
        samples.append(("Correct action=Tell Teammate", tell[0]))

    for label, sample in samples:
        lines.append(f"--- {label} (ID={sample['scenario_id']}, Extra={sample['extra']}) ---")
        lines.append(f"Events: {sample['event_count']}, Words: {sample['word_count']}")
        lines.append(f"KS: self={sample['ks_self']}, teammate={sample['ks_teammate']}, opponent={sample['ks_opponent']}")
        lines.append(f"Action: {sample['correct_action']}")
        lines.append(f"Description:")
        lines.append(f"  {sample['scenario_desc']}")
        lines.append("")

    # =========================================================================
    # 8. Summary of Findings
    # =========================================================================
    lines.append("=" * 80)
    lines.append("8. SUMMARY")
    lines.append("=" * 80)

    lines.append(f"\nOverall: {len(parsed_data)} scenarios across {len(file_paths)} runs.")
    lines.append(f"Extra=0 avg events: {statistics.mean(e0_counts):.1f} (range {min(e0_counts)}-{max(e0_counts)})")
    lines.append(f"Extra=1 avg events: {statistics.mean(e1_counts):.1f} (range {min(e1_counts)}-{max(e1_counts)})")
    lines.append(f"Average Extra delta: +{statistics.mean(e1_counts) - statistics.mean(e0_counts):.1f} events")

    if failures:
        lines.append(f"\nGeneration issues:")
        lines.append(f"  {len(failures)} scenario(s) with Extra=1 mean delta ≤ 0: IDs {failures}")
    if run_failures:
        lines.append(f"  {len(run_failures)} individual run cases where Extra=1 ≤ Extra=0")

    if high_cv:
        lines.append(f"  {len(high_cv)} scenario(s) with high event count variance (CV > 0.3)")

    if not failures and not run_failures and not high_cv:
        lines.append("\nNo generation issues detected.")

    return "\n".join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'tom_llm_logs')
    output_path = os.path.join(logs_dir, 'scenario_analysis.txt')

    file_paths = find_recent_game_data(logs_dir, n=8)
    if not file_paths:
        print("No game_data.json files found.")
        return

    print(f"Loading {len(file_paths)} most recent game_data.json files...")
    parsed_data = load_and_parse_all(file_paths)
    print(f"Parsed {len(parsed_data)} scenario records.")

    stats = compute_per_scenario_stats(parsed_data)
    report = generate_report(parsed_data, stats, file_paths)

    print(report)

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")


if __name__ == '__main__':
    main()
