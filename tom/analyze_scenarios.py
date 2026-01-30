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
                'extra': normalize_extra(extra_val),
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

    extra0a = [p for p in parsed_data if p['extra'] == '0A']
    extra0b = [p for p in parsed_data if p['extra'] == '0B']
    extra1a = [p for p in parsed_data if p['extra'] == '1A']
    extra1b = [p for p in parsed_data if p['extra'] == '1B']
    lines.append(f"  Extra=0A: {len(extra0a)}")
    lines.append(f"  Extra=0B: {len(extra0b)}")
    lines.append(f"  Extra=1A: {len(extra1a)}")
    lines.append(f"  Extra=1B: {len(extra1b)}")

    # =========================================================================
    # 1. Event Count Summary Table
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("1. EVENT COUNT PER SCENARIO (across runs)")
    lines.append("=" * 80)
    lines.append(f"\n{'ID':>4} {'Extra':>5} {'Mean':>6} {'StDev':>6} {'Min':>4} {'Max':>4} {'N':>3} {'Action':>15} {'KS_Self':>15}")
    lines.append("-" * 72)

    # Sort by scenario ID (numeric) then extra (0A, 0B, 1A, 1B)
    extra_order = {'0A': 0, '0B': 1, '1A': 2, '1B': 3}
    sorted_keys = sorted(stats.keys(), key=lambda k: (int(k[0]), extra_order.get(k[1], 99)))
    for key in sorted_keys:
        s = stats[key]
        sid, extra = key
        lines.append(
            f"{sid:>4} {extra:>5} {s['event_mean']:>6.1f} {s['event_stdev']:>6.2f} "
            f"{s['event_min']:>4} {s['event_max']:>4} {s['n_runs']:>3} "
            f"{s['correct_action']:>15} {s['ks_self']:>15}"
        )

    # =========================================================================
    # 2. Extra=1A vs Extra=1B Complexity Comparison (ECT effect)
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("2. EXTRA=1A vs EXTRA=1B COMPLEXITY (ECT effect)")
    lines.append("=" * 80)

    e1a_counts = [p['event_count'] for p in extra1a]
    e1b_counts = [p['event_count'] for p in extra1b]

    if e1a_counts and e1b_counts:
        lines.append(f"\nExtra=1A: mean={statistics.mean(e1a_counts):.2f}, "
                     f"stdev={statistics.stdev(e1a_counts):.2f}, "
                     f"min={min(e1a_counts)}, max={max(e1a_counts)}")
        lines.append(f"Extra=1B: mean={statistics.mean(e1b_counts):.2f}, "
                     f"stdev={statistics.stdev(e1b_counts):.2f}, "
                     f"min={min(e1b_counts)}, max={max(e1b_counts)}")
        lines.append(f"Delta (Extra=1B - Extra=1A): {statistics.mean(e1b_counts) - statistics.mean(e1a_counts):.2f} events on average")
    else:
        lines.append("\nInsufficient data for 1A vs 1B comparison.")

    # Per-scenario delta for 1A vs 1B
    lines.append(f"\n{'ID':>4} {'1A Mean':>8} {'1B Mean':>8} {'Delta':>7} {'Flag':>6}")
    lines.append("-" * 40)

    failures = []
    for sid in sorted(set(k[0] for k in stats.keys()), key=int):
        e1a_key = (sid, '1A')
        e1b_key = (sid, '1B')
        if e1a_key in stats and e1b_key in stats:
            e1a_mean = stats[e1a_key]['event_mean']
            e1b_mean = stats[e1b_key]['event_mean']
            delta = e1b_mean - e1a_mean
            flag = "WARN" if delta <= 0 else ""
            lines.append(f"{sid:>4} {e1a_mean:>8.1f} {e1b_mean:>8.1f} {delta:>7.1f} {flag:>6}")
            if delta <= 0:
                failures.append(sid)

    if failures:
        lines.append(f"\nWARNING: {len(failures)} scenario(s) where Extra=1B has ≤ events than Extra=1A: {failures}")
    elif e1a_counts and e1b_counts:
        lines.append("\nAll scenarios have more events in Extra=1B than Extra=1A.")

    # Check individual runs for any Extra=1B <= Extra=1A
    run_failures = []
    for sid in sorted(set(k[0] for k in stats.keys()), key=int):
        e1a_key = (sid, '1A')
        e1b_key = (sid, '1B')
        if e1a_key in stats and e1b_key in stats:
            e1a_counts_list = stats[e1a_key]['event_counts']
            e1b_counts_list = stats[e1b_key]['event_counts']
            for i, (e1a_c, e1b_c) in enumerate(zip(e1a_counts_list, e1b_counts_list)):
                if e1b_c <= e1a_c:
                    run_failures.append((sid, i, e1a_c, e1b_c))

    if run_failures:
        lines.append(f"\nIndividual run failures (Extra=1B ≤ Extra=1A): {len(run_failures)}")
        for sid, run_idx, e1a_c, e1b_c in run_failures[:20]:
            lines.append(f"  ID {sid}, run {run_idx}: Extra=1A={e1a_c}, Extra=1B={e1b_c}")
    elif e1a_counts and e1b_counts:
        lines.append("No individual run failures (Extra=1B always > Extra=1A in every run).")

    # =========================================================================
    # 2b. Extra=0A vs Extra=0B Complexity Comparison (SIT effect)
    # =========================================================================
    lines.append("\n" + "-" * 80)
    lines.append("2b. EXTRA=0A vs EXTRA=0B COMPLEXITY (SIT effect)")
    lines.append("-" * 80)

    e0a_counts = [p['event_count'] for p in extra0a]
    e0b_counts = [p['event_count'] for p in extra0b]

    if e0a_counts and e0b_counts:
        lines.append(f"\nExtra=0A: mean={statistics.mean(e0a_counts):.2f}, "
                     f"stdev={statistics.stdev(e0a_counts):.2f}, "
                     f"min={min(e0a_counts)}, max={max(e0a_counts)}")
        lines.append(f"Extra=0B: mean={statistics.mean(e0b_counts):.2f}, "
                     f"stdev={statistics.stdev(e0b_counts):.2f}, "
                     f"min={min(e0b_counts)}, max={max(e0b_counts)}")
        lines.append(f"Delta (Extra=0B - Extra=0A): {statistics.mean(e0b_counts) - statistics.mean(e0a_counts):.2f} events on average")
    else:
        lines.append("\nNo Extra=0A/0B data found (new scenario type).")

    # =========================================================================
    # 3. Event Type Distribution
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("3. EVENT TYPE DISTRIBUTION")
    lines.append("=" * 80)

    type_totals = {
        'ALL': defaultdict(int),
        'Extra=0A': defaultdict(int),
        'Extra=0B': defaultdict(int),
        'Extra=1A': defaultdict(int),
        'Extra=1B': defaultdict(int)
    }
    for p in parsed_data:
        group = f"Extra={p['extra']}"
        for etype, count in p['event_types'].items():
            type_totals['ALL'][etype] += count
            if group in type_totals:
                type_totals[group][etype] += count

    all_types = sorted(set().union(*[t.keys() for t in type_totals.values()]))
    lines.append(f"\n{'Type':>10} {'ALL':>8} {'0A':>8} {'0B':>8} {'1A':>8} {'1B':>8}")
    lines.append("-" * 54)
    for etype in all_types:
        all_c = type_totals['ALL'].get(etype, 0)
        e0a_c = type_totals['Extra=0A'].get(etype, 0)
        e0b_c = type_totals['Extra=0B'].get(etype, 0)
        e1a_c = type_totals['Extra=1A'].get(etype, 0)
        e1b_c = type_totals['Extra=1B'].get(etype, 0)
        lines.append(f"{etype:>10} {all_c:>8} {e0a_c:>8} {e0b_c:>8} {e1a_c:>8} {e1b_c:>8}")

    total_all = sum(type_totals['ALL'].values())
    total_e0a = sum(type_totals['Extra=0A'].values())
    total_e0b = sum(type_totals['Extra=0B'].values())
    total_e1a = sum(type_totals['Extra=1A'].values())
    total_e1b = sum(type_totals['Extra=1B'].values())
    lines.append(f"{'TOTAL':>10} {total_all:>8} {total_e0a:>8} {total_e0b:>8} {total_e1a:>8} {total_e1b:>8}")

    # Proportions
    lines.append(f"\nProportions:")
    lines.append(f"{'Type':>10} {'ALL':>8} {'0A':>8} {'0B':>8} {'1A':>8} {'1B':>8}")
    lines.append("-" * 54)
    for etype in all_types:
        all_pct = type_totals['ALL'].get(etype, 0) / total_all * 100 if total_all else 0
        e0a_pct = type_totals['Extra=0A'].get(etype, 0) / total_e0a * 100 if total_e0a else 0
        e0b_pct = type_totals['Extra=0B'].get(etype, 0) / total_e0b * 100 if total_e0b else 0
        e1a_pct = type_totals['Extra=1A'].get(etype, 0) / total_e1a * 100 if total_e1a else 0
        e1b_pct = type_totals['Extra=1B'].get(etype, 0) / total_e1b * 100 if total_e1b else 0
        lines.append(f"{etype:>10} {all_pct:>7.1f}% {e0a_pct:>7.1f}% {e0b_pct:>7.1f}% {e1a_pct:>7.1f}% {e1b_pct:>7.1f}%")

    # =========================================================================
    # 4. Description Length
    # =========================================================================
    lines.append("\n" + "=" * 80)
    lines.append("4. DESCRIPTION LENGTH")
    lines.append("=" * 80)

    for label, subset in [("Extra=0A", extra0a), ("Extra=0B", extra0b), ("Extra=1A", extra1a), ("Extra=1B", extra1b), ("ALL", parsed_data)]:
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

    for extra_level in ['0A', '0B', '1A', '1B']:
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

    # Simple Extra=0A (minimal filler)
    simple_0a = [p for p in recent if p['extra'] == '0A' and p['event_count'] <= 3]
    if simple_0a:
        samples.append(("Simple Extra=0A", simple_0a[0]))

    # Simple Extra=1A (filler for SIT parity)
    simple_1a = [p for p in recent if p['extra'] == '1A' and p['event_count'] <= 3]
    if simple_1a:
        samples.append(("Simple Extra=1A", simple_1a[0]))

    # Complex Extra=1B (ECT events)
    complex_ = sorted([p for p in recent if p['extra'] == '1B'], key=lambda x: -x['event_count'])
    if complex_:
        samples.append(("Complex Extra=1B (most events)", complex_[0]))

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
    if e1a_counts:
        lines.append(f"Extra=1A avg events: {statistics.mean(e1a_counts):.1f} (range {min(e1a_counts)}-{max(e1a_counts)})")
    if e1b_counts:
        lines.append(f"Extra=1B avg events: {statistics.mean(e1b_counts):.1f} (range {min(e1b_counts)}-{max(e1b_counts)})")
    if e1a_counts and e1b_counts:
        lines.append(f"Average 1A→1B delta: +{statistics.mean(e1b_counts) - statistics.mean(e1a_counts):.1f} events (ECT effect)")
    if e0a_counts and e0b_counts:
        lines.append(f"Extra=0A avg events: {statistics.mean(e0a_counts):.1f} (range {min(e0a_counts)}-{max(e0a_counts)})")
        lines.append(f"Extra=0B avg events: {statistics.mean(e0b_counts):.1f} (range {min(e0b_counts)}-{max(e0b_counts)})")
        lines.append(f"Average 0A→0B delta: +{statistics.mean(e0b_counts) - statistics.mean(e0a_counts):.1f} events (SIT effect)")

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
