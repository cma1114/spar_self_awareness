#!/usr/bin/env python3
"""
Analyze ToM test results by history mode and extra condition.

Reads game_data.json files from a specified directory (and subdirectories),
aggregates by model/history_mode and extra condition, and produces:
1. Overall performance by model/history_mode and extra condition
2. Mastery scores by model/history_mode (aggregated across extra)
3. Mastery scores by extra condition (aggregated across history_mode, per model)
"""

import json
import glob
import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Wilson CI for error bars
# =============================================================================

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return (lower, upper) bounds of the 95% Wilson CI."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    half_width = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (centre - half_width) / denom
    upper = (centre + half_width) / denom
    return lower, upper


# =============================================================================
# ToM Mastery Categories (copied from analyze_results.py)
# =============================================================================

TOM_MASTERY_CATEGORIES = {
    'self_knowledge_belief': {
        'name': 'Self: Knowledge vs Belief',
        'short_name': 'Self K/B',
        'description': 'Distinguishing own knowledge from mere belief',
        'components': [
            {'scenarios': [7, 8, 9], 'action': 'Pass', 'weight': 2},
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 3},
        ],
    },
    'teammate_knowledge_belief': {
        'name': 'Teammate: Knowledge vs Belief',
        'short_name': 'Teammate K/B',
        'description': 'Distinguishing teammate knowledge from belief',
        'components': [
            {'scenarios': [20, 21, 22], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
    'combined_uncertainty': {
        'name': 'Combined Uncertainty',
        'short_name': 'Combined Unc.',
        'description': 'Handling self + teammate uncertainty together',
        'components': [
            {'scenarios': [10, 11, 23, 24], 'action': 'Pass', 'weight': 1},
        ],
    },
    'true_false_belief': {
        'name': 'True vs False Belief',
        'short_name': 'True/False',
        'description': 'Distinguishing true belief from false belief',
        'components': [
            {'scenarios': [14, 15, 16], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
    'teammate_opponent': {
        'name': 'Teammate vs Opponent',
        'short_name': 'Team/Opp',
        'description': 'Treating teammate differently from opponent',
        'components': [
            {'scenarios': [12, 13], 'action': 'Ask', 'weight': 1},
            {'scenarios': [37, 39], 'action': 'Pass', 'weight': 1},
            {'scenarios': [30, 31, 32], 'action': 'Pass', 'weight': 1},
            {'scenarios': [17, 18, 19], 'action': 'Tell', 'weight': 1},
        ],
    },
}


# =============================================================================
# History mode label mapping
# =============================================================================

HISTORY_MODE_LABELS = {
    'none': 'no history',
    'no reasoning': 'scenario history only',
    'no_reasoning': 'scenario history only',
    'with reasoning': 'reasoning history',
    'with_reasoning': 'reasoning history',
}

HISTORY_MODE_SHORT = {
    'none': 'none',
    'no reasoning': 'scenario',
    'no_reasoning': 'scenario',
    'with reasoning': 'reasoning',
    'with_reasoning': 'reasoning',
}

# Canonical order for history modes (left to right in graphs, top to bottom in legends)
HISTORY_MODE_ORDER = ['none', 'no_reasoning', 'no reasoning', 'with_reasoning', 'with reasoning']


def get_history_sort_key(history_mode: str) -> int:
    """Return sort key for history mode to ensure consistent ordering."""
    try:
        return HISTORY_MODE_ORDER.index(history_mode)
    except ValueError:
        return len(HISTORY_MODE_ORDER)  # Unknown modes go last


def sort_history_modes(history_modes: List[str]) -> List[str]:
    """Sort history modes in preferred order."""
    return sorted(history_modes, key=get_history_sort_key)


def get_history_label(raw_mode: str, short: bool = False) -> str:
    """Convert raw history_mode value to display label."""
    if short:
        return HISTORY_MODE_SHORT.get(raw_mode, raw_mode)
    return HISTORY_MODE_LABELS.get(raw_mode, raw_mode)


# =============================================================================
# Model name prettification
# =============================================================================

def prettify_model_name(raw_name: str) -> str:
    """Convert raw model name to a more readable format."""
    name = raw_name
    
    # Remove common prefixes
    prefixes_to_remove = [
        'anthropic-', 'claude-', 'openai-', 'google-', 'meta-', 
        'deepseek-', 'mistralai-', 'qwen-'
    ]
    for prefix in prefixes_to_remove:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
    
    # Replace underscores and hyphens with spaces for processing
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Common replacements for readability
    replacements = [
        ('sonnet 4 5', 'Sonnet 4.5'),
        ('sonnet 4', 'Sonnet 4'),
        ('opus 4 5', 'Opus 4.5'),
        ('opus 4', 'Opus 4'),
        ('haiku 4 5', 'Haiku 4.5'),
        ('haiku 4', 'Haiku 4'),
        ('gpt 5', 'GPT-5'),
        ('gpt 4', 'GPT-4'),
        ('gpt 4o', 'GPT-4o'),
        ('o3', 'o3'),
        ('o1', 'o1'),
        ('llama 3 1', 'Llama 3.1'),
        ('llama 3', 'Llama 3'),
        ('qwen 3', 'Qwen 3'),
        ('qwen 2 5', 'Qwen 2.5'),
        ('gemini 2 5', 'Gemini 2.5'),
        ('gemini 2', 'Gemini 2'),
        ('deepseek v3', 'DeepSeek V3'),
        ('deepseek r1', 'DeepSeek R1'),
        ('405b', '405B'),
        ('70b', '70B'),
        ('8b', '8B'),
        (' think', ' think'),  # Preserve 'think' indicator
    ]
    
    name_lower = name.lower()
    for old, new in replacements:
        if old in name_lower:
            # Find position and replace preserving case logic
            idx = name_lower.find(old)
            name = name[:idx] + new + name[idx + len(old):]
            name_lower = name.lower()
    
    # Clean up extra spaces
    name = ' '.join(name.split())
    
    # Capitalize first letter if not already
    if name and name[0].islower():
        name = name[0].upper() + name[1:]
    
    return name


def extract_model_name(filepath: str) -> str:
    """Extract model name from filename, stripping timestamp."""
    basename = os.path.basename(filepath)
    # Pattern: {model}_{timestamp}_game_data.json
    match = re.match(r'(.+?)_\d+_game_data\.json$', basename)
    if match:
        return match.group(1)
    return basename


# =============================================================================
# Data loading
# =============================================================================

def load_game_data(filepath: str) -> List[dict]:
    """Load and return turn records from a game_data.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # New format has 'turn_records' key
    if isinstance(data, dict) and 'turn_records' in data:
        return data['turn_records']
    
    # Fallback for old format (list of records)
    if isinstance(data, list):
        return data
    
    return []


def filter_player_a(records: List[dict]) -> List[dict]:
    """Filter to only player A's records."""
    return [r for r in records if r.get('character') == 'A']


def find_json_files(base_dir: str) -> List[str]:
    """Find all game_data.json files recursively."""
    pattern = os.path.join(base_dir, '**', '*_game_data.json')
    return glob.glob(pattern, recursive=True)


# =============================================================================
# Statistics computation
# =============================================================================

def compute_accuracy_stats(records: List[dict]) -> dict:
    """Compute accuracy and CI for a set of records."""
    n_total = len(records)
    n_optimal = sum(1 for r in records if r.get('was_optimal'))
    
    rate = n_optimal / n_total if n_total > 0 else 0
    ci = wilson_ci(n_optimal, n_total)
    
    return {
        'n': n_total,
        'k': n_optimal,
        'rate': rate,
        'ci': ci,
        'ci_half_width': (ci[1] - ci[0]) / 2,
    }


def compute_mastery_score(records: List[dict], category: dict) -> dict:
    """
    Compute mastery score for a single ToM category.
    
    Returns dict with:
    - score: weighted accuracy (0-1)
    - n: total weighted trials
    - k: weighted correct
    - ci: Wilson confidence interval
    """
    total_weighted = 0
    correct_weighted = 0
    
    for component in category['components']:
        scenarios = component['scenarios']
        weight = component['weight']
        
        # Filter records for these scenarios
        scenario_strs = [str(s) for s in scenarios]
        component_records = [r for r in records if r.get('scenario_id') in scenario_strs]
        
        n_total = len(component_records)
        n_correct = sum(1 for r in component_records if r.get('was_optimal'))
        
        total_weighted += n_total * weight
        correct_weighted += n_correct * weight
    
    score = correct_weighted / total_weighted if total_weighted > 0 else 0
    
    # For CI, use unweighted counts (approximate)
    unweighted_n = sum(
        len([r for r in records if r.get('scenario_id') in [str(s) for s in comp['scenarios']]])
        for comp in category['components']
    )
    unweighted_k = sum(
        sum(1 for r in records if r.get('scenario_id') in [str(s) for s in comp['scenarios']] and r.get('was_optimal'))
        for comp in category['components']
    )
    ci = wilson_ci(unweighted_k, unweighted_n)
    
    return {
        'score': score,
        'n': total_weighted,
        'k': correct_weighted,
        'ci': ci,
        'ci_half_width': (ci[1] - ci[0]) / 2,
    }


def compute_all_mastery_scores(records: List[dict]) -> dict:
    """Compute all ToM mastery scores for a set of records."""
    results = {}
    for key, category in TOM_MASTERY_CATEGORIES.items():
        results[key] = compute_mastery_score(records, category)
        results[key]['name'] = category['name']
        results[key]['short_name'] = category['short_name']
    return results


# =============================================================================
# Analysis 1: Overall performance by model/history_mode and extra condition
# =============================================================================

def analysis_overall_performance(
    model_history_records: Dict[Tuple[str, str], List[dict]],
    output_dir: str
):
    """
    Analyze overall performance across all scenarios,
    by model/history_mode and extra condition.
    """
    # Collect all model/history combinations, sorted by model then history mode order
    model_history_keys = sorted(
        model_history_records.keys(),
        key=lambda x: (x[0], get_history_sort_key(x[1]))
    )
    
    # Collect all extra conditions
    all_extras = set()
    for records in model_history_records.values():
        for r in records:
            extra = r.get('extra', '0A')
            if extra:
                all_extras.add(extra)
    extra_conditions = sorted(all_extras)
    
    # Compute stats for each combination
    results = {}
    for mh_key in model_history_keys:
        records = model_history_records[mh_key]
        model_name, history_mode = mh_key
        
        results[mh_key] = {
            'overall': compute_accuracy_stats(records),
            'by_extra': {}
        }
        
        for extra in extra_conditions:
            extra_records = [r for r in records if r.get('extra') == extra]
            results[mh_key]['by_extra'][extra] = compute_accuracy_stats(extra_records)
    
    # Generate text output
    lines = []
    lines.append("=" * 100)
    lines.append("OVERALL PERFORMANCE BY MODEL/HISTORY MODE AND EXTRA CONDITION")
    lines.append("=" * 100)
    
    # Header
    header = f"{'Model / History Mode':<40}  {'N':>6}  {'Overall':>8}"
    for extra in extra_conditions:
        header += f"  {extra:>8}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Data rows
    for mh_key in model_history_keys:
        model_name, history_mode = mh_key
        pretty_model = prettify_model_name(model_name)
        history_label = get_history_label(history_mode, short=True)
        display_name = f"{pretty_model} / {history_label}"
        
        stats = results[mh_key]
        overall = stats['overall']
        
        row = f"{display_name:<40}  {overall['n']:>6}  {overall['rate']*100:>7.1f}%"
        
        for extra in extra_conditions:
            extra_stats = stats['by_extra'].get(extra, {'rate': 0, 'n': 0})
            if extra_stats['n'] > 0:
                row += f"  {extra_stats['rate']*100:>7.1f}%"
            else:
                row += f"  {'N/A':>8}"
        
        lines.append(row)
    
    lines.append("=" * 100)
    
    # Save text file
    text_path = os.path.join(output_dir, 'performance_by_history_extra.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {text_path}")
    
    # Generate chart - one subplot per model
    # x-axis is extra conditions, grouped bars are history modes
    
    # Aggregate by model to find which history modes each model has
    model_histories: Dict[str, List[str]] = defaultdict(list)
    for model_name, history_mode in model_history_keys:
        if history_mode not in model_histories[model_name]:
            model_histories[model_name].append(history_mode)
    
    models = sorted(model_histories.keys())
    n_models = len(models)
    
    if n_models == 0:
        print("No models to plot")
        return
    
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    
    # Consistent colors for history modes across all subplots
    all_history_modes = sort_history_modes(list(set(hm for hm_list in model_histories.values() for hm in hm_list)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_history_modes)))
    color_map = {hm: colors[i] for i, hm in enumerate(all_history_modes)}
    
    for idx, model in enumerate(models):
        ax = axes[idx // n_cols, idx % n_cols]
        pretty_model = prettify_model_name(model)
        
        history_modes = sort_history_modes(model_histories[model])
        
        x = np.arange(len(extra_conditions))
        n_histories = len(history_modes)
        width = 0.8 / max(n_histories, 1)
        
        for i, history_mode in enumerate(history_modes):
            mh_key = (model, history_mode)
            
            rates = []
            errs = []
            for extra in extra_conditions:
                stats = results[mh_key]['by_extra'].get(extra, {'rate': 0, 'ci_half_width': 0, 'n': 0})
                rates.append(stats['rate'] * 100 if stats['n'] > 0 else 0)
                errs.append(stats['ci_half_width'] * 100 if stats['n'] > 0 else 0)
            
            offset = width * (i - n_histories / 2 + 0.5)
            hist_label = get_history_label(history_mode)
            ax.bar(x + offset, rates, width, 
                   label=hist_label if idx == 0 else "",
                   yerr=errs, capsize=2, color=color_map[history_mode], edgecolor='white')
        
        ax.set_ylabel('Optimal Action Rate (%)', fontsize=10)
        ax.set_title(pretty_model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Extra={e}' for e in extra_conditions], fontsize=9)
        ax.set_xlabel('Extra Condition', fontsize=10)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', fontsize=9, title='History Mode')
    
    fig.suptitle('Performance by History Mode and Extra Condition (per Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'performance_by_history_extra.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart_path}")
    plt.close(fig)
    
    return results


# =============================================================================
# Analysis 2: Mastery by model/history_mode (aggregated across extra)
# =============================================================================

def analysis_mastery_by_history(
    model_history_records: Dict[Tuple[str, str], List[dict]],
    output_dir: str
):
    """
    Analyze mastery scores by model/history_mode,
    aggregated across extra conditions.
    """
    model_history_keys = sorted(
        model_history_records.keys(),
        key=lambda x: (x[0], get_history_sort_key(x[1]))
    )
    cat_keys = list(TOM_MASTERY_CATEGORIES.keys())
    
    # Compute mastery for each model/history combination
    results = {}
    for mh_key in model_history_keys:
        records = model_history_records[mh_key]
        results[mh_key] = compute_all_mastery_scores(records)
    
    # Aggregate by model (to find which history modes each model has)
    model_histories: Dict[str, List[str]] = defaultdict(list)
    for model_name, history_mode in model_history_keys:
        model_histories[model_name].append(history_mode)
    
    models = sorted(model_histories.keys())
    
    # Generate text output
    lines = []
    lines.append("=" * 115)
    lines.append("MASTERY SCORES BY MODEL/HISTORY MODE (aggregated across Extra conditions)")
    lines.append("=" * 115)
    
    # Header
    header = f"{'Model / History Mode':<40}"
    for key in cat_keys:
        short_name = TOM_MASTERY_CATEGORIES[key]['short_name']
        header += f"  {short_name:>13}"
    lines.append(header)
    lines.append("-" * 115)
    
    # Data rows
    for mh_key in model_history_keys:
        model_name, history_mode = mh_key
        pretty_model = prettify_model_name(model_name)
        history_label = get_history_label(history_mode, short=True)
        display_name = f"{pretty_model} / {history_label}"
        
        row = f"{display_name:<40}"
        for key in cat_keys:
            score = results[mh_key][key]['score'] * 100
            row += f"  {score:>12.1f}%"
        lines.append(row)
    
    lines.append("=" * 115)
    
    # Save text file
    text_path = os.path.join(output_dir, 'mastery_by_history.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {text_path}")
    
    # Generate chart - one subplot per model
    # x-axis is mastery categories, grouped bars are history modes
    n_models = len(models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    
    # Get all unique history modes for consistent coloring
    all_history_modes = sort_history_modes(list(set(hm for hm_list in model_histories.values() for hm in hm_list)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_history_modes)))
    color_map = {hm: colors[i] for i, hm in enumerate(all_history_modes)}
    
    for idx, model in enumerate(models):
        ax = axes[idx // n_cols, idx % n_cols]
        pretty_model = prettify_model_name(model)
        
        x = np.arange(len(cat_keys))
        history_modes = model_histories[model]
        n_histories = len(history_modes)
        width = 0.8 / max(n_histories, 1)
        
        for i, history_mode in enumerate(sort_history_modes(history_modes)):
            mh_key = (model, history_mode)
            
            scores = [results[mh_key][cat_key]['score'] * 100 for cat_key in cat_keys]
            errs = [results[mh_key][cat_key]['ci_half_width'] * 100 for cat_key in cat_keys]
            
            offset = width * (i - n_histories / 2 + 0.5)
            label = get_history_label(history_mode, short=False)
            ax.bar(x + offset, scores, width, label=label,
                   yerr=errs, capsize=2, color=color_map[history_mode], edgecolor='white')
        
        ax.set_ylabel('Mastery Score (%)', fontsize=10)
        ax.set_title(pretty_model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        xlabels = [TOM_MASTERY_CATEGORIES[key]['short_name'] for key in cat_keys]
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel('Mastery Category', fontsize=10)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, title='History Mode')
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    fig.suptitle('Mastery Scores by History Mode (per Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'mastery_by_history.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart_path}")
    plt.close(fig)
    
    return results


# =============================================================================
# Analysis 3: Mastery by extra condition (aggregated across history_mode, per model)
# =============================================================================

def analysis_mastery_by_extra(
    model_history_records: Dict[Tuple[str, str], List[dict]],
    output_dir: str
):
    """
    Analyze mastery scores by extra condition,
    aggregated across history_mode but per model.
    """
    # Aggregate records by model (across history modes)
    model_records: Dict[str, List[dict]] = defaultdict(list)
    for (model_name, history_mode), records in model_history_records.items():
        model_records[model_name].extend(records)
    
    models = sorted(model_records.keys())
    cat_keys = list(TOM_MASTERY_CATEGORIES.keys())
    
    # Collect all extra conditions
    all_extras = set()
    for records in model_records.values():
        for r in records:
            extra = r.get('extra', '0A')
            if extra:
                all_extras.add(extra)
    extra_conditions = sorted(all_extras)
    
    # Compute mastery for each model × extra combination
    results = {}
    for model in models:
        results[model] = {}
        for extra in extra_conditions:
            extra_records = [r for r in model_records[model] if r.get('extra') == extra]
            if extra_records:
                results[model][extra] = compute_all_mastery_scores(extra_records)
            else:
                results[model][extra] = None
    
    # Generate text output
    lines = []
    lines.append("=" * 85)
    lines.append("MASTERY SCORES BY EXTRA CONDITION (aggregated across History Mode, per Model)")
    lines.append("=" * 85)
    
    for model in models:
        pretty_model = prettify_model_name(model)
        lines.append(f"\n{pretty_model}")
        lines.append("-" * 85)
        
        # Header for this model
        header = f"{'Extra':<8}"
        for key in cat_keys:
            short_name = TOM_MASTERY_CATEGORIES[key]['short_name']
            header += f"  {short_name:>13}"
        lines.append(header)
        lines.append("-" * 85)
        
        # Data rows for each extra condition
        for extra in extra_conditions:
            if results[model][extra] is None:
                row = f"{extra:<8}"
                for _ in cat_keys:
                    row += f"  {'N/A':>13}"
            else:
                row = f"{extra:<8}"
                for key in cat_keys:
                    score = results[model][extra][key]['score'] * 100
                    row += f"  {score:>12.1f}%"
            lines.append(row)
    
    lines.append("\n" + "=" * 85)
    
    # Save text file
    text_path = os.path.join(output_dir, 'mastery_by_extra.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {text_path}")
    
    # Generate chart - one subplot per model
    # x-axis is mastery categories, grouped bars are extra conditions
    n_models = len(models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), squeeze=False)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(extra_conditions)))
    
    for idx, model in enumerate(models):
        ax = axes[idx // n_cols, idx % n_cols]
        pretty_model = prettify_model_name(model)
        
        x = np.arange(len(cat_keys))
        n_extras = len(extra_conditions)
        width = 0.8 / max(n_extras, 1)
        
        for i, extra in enumerate(extra_conditions):
            scores = []
            errs = []
            for cat_key in cat_keys:
                if results[model][extra] is not None:
                    scores.append(results[model][extra][cat_key]['score'] * 100)
                    errs.append(results[model][extra][cat_key]['ci_half_width'] * 100)
                else:
                    scores.append(0)
                    errs.append(0)
            
            offset = width * (i - n_extras / 2 + 0.5)
            ax.bar(x + offset, scores, width, label=f'Extra={extra}' if idx == 0 else "",
                   yerr=errs, capsize=2, color=colors[i], edgecolor='white')
        
        ax.set_ylabel('Mastery Score (%)', fontsize=10)
        ax.set_title(pretty_model, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        xlabels = [TOM_MASTERY_CATEGORIES[key]['short_name'] for key in cat_keys]
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel('Mastery Category', fontsize=10)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', fontsize=9)
    
    fig.suptitle('Mastery Scores by Extra Condition (per Model)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, 'mastery_by_extra.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {chart_path}")
    plt.close(fig)
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze ToM test results by history mode and extra condition.'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing game_data.json files (searches recursively)'
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    # Find all JSON files
    json_files = find_json_files(args.directory)
    print(f"Found {len(json_files)} game_data.json files")
    
    if not json_files:
        print("No game_data.json files found.")
        return
    
    # Load and organize records by (model, history_mode)
    model_history_records: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    
    for filepath in json_files:
        records = load_game_data(filepath)
        a_records = filter_player_a(records)
        
        if not a_records:
            print(f"  Skipping {filepath}: no player A records")
            continue
        
        model_name = extract_model_name(filepath)
        
        # Group by history_mode within this file
        for record in a_records:
            history_mode = record.get('history_mode', 'none')
            key = (model_name, history_mode)
            model_history_records[key].append(record)
    
    if not model_history_records:
        print("No valid records found after filtering.")
        return
    
    print(f"\nFound {len(model_history_records)} model/history_mode combinations:")
    for (model, hist), records in sorted(
        model_history_records.items(),
        key=lambda x: (x[0][0], get_history_sort_key(x[0][1]))
    ):
        pretty = prettify_model_name(model)
        hist_label = get_history_label(hist)
        print(f"  {pretty} / {hist_label}: {len(records)} records")
    
    # Run analyses
    print("\n" + "=" * 60)
    print("Running Analysis 1: Overall Performance")
    print("=" * 60)
    analysis_overall_performance(model_history_records, args.directory)
    
    print("\n" + "=" * 60)
    print("Running Analysis 2: Mastery by History Mode")
    print("=" * 60)
    analysis_mastery_by_history(model_history_records, args.directory)
    
    print("\n" + "=" * 60)
    print("Running Analysis 3: Mastery by Extra Condition")
    print("=" * 60)
    analysis_mastery_by_extra(model_history_records, args.directory)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
