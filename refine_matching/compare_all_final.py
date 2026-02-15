#!/usr/bin/env python3
"""
Comprehensive comparison of ALL methods: individual models and ensembles.

Usage:
    python compare_all_final.py
"""

import csv
import math
from pathlib import Path


def load_csv(path):
    """Load CSV into dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def evaluate_predictions(pred, gt):
    """Evaluate predictions and return score and accuracy breakdown."""
    common_ids = sorted(set(gt) & set(pred))
    if len(common_ids) == 0:
        return None

    distances = []
    for i in common_ids:
        dx = pred[i][0] - gt[i][0]
        dy = pred[i][1] - gt[i][1]
        distances.append(math.sqrt(dx * dx + dy * dy))

    thresholds = [(25, "5m"), (125, "25m"), (500, "100m")]
    accuracies = {}
    for px_thresh, label in thresholds:
        acc = sum(1 for d in distances if d <= px_thresh) / len(distances) * 100
        accuracies[label] = acc

    score = sum(accuracies.values()) / len(accuracies)

    distances.sort()
    stats = {
        'median': distances[len(distances)//2],
        'mean': sum(distances)/len(distances),
        'n_samples': len(common_ids)
    }

    return score, accuracies, stats


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    print("="*90)
    print("COMPREHENSIVE COMPARISON: ALL INDIVIDUAL MODELS + ALL ENSEMBLES")
    print("="*90)

    # Load ground truth
    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    gt = load_csv(gt_path)

    results = []

    # Individual models
    individual_models = [
        'xoftr', 'edm', 'superglue', 'rdd-lightglue',
        'rdd-star', 'dedode', 'affine-steerers',
        'liftfeat', 'loftr', 'minima-loftr', 'minima-roma',
        'ripe', 'se2loftr',
        'minima-roma_y-1', 'minima-roma_y-2', 'minima-roma_y-3',
        'minima-roma_y-4', 'minima-roma_y-5', 'minima-roma_y-6',
        'minima-roma_y-7', 'minima-roma_y-8',
        'se2loftr_y-3', 'se2loftr_y+3'
    ]

    print("\nEvaluating individual models:")
    for model in individual_models:
        pred_path = root / f"train_predictions_{model}.csv"
        if pred_path.exists():
            pred = load_csv(pred_path)
            result = evaluate_predictions(pred, gt)
            if result:
                score, accuracies, stats = result
                results.append({
                    'name': model,
                    'type': 'individual',
                    'score': score,
                    'acc_5m': accuracies['5m'],
                    'acc_25m': accuracies['25m'],
                    'acc_100m': accuracies['100m'],
                    'median': stats['median'],
                    'mean': stats['mean'],
                    'n_samples': stats['n_samples']
                })
                print(f"  ✓ {model}")

    # Ensemble predictions - check what exists
    ensemble_files = [
        ('ensemble_median', 'Top3 Median (simple)'),
        ('ensemble_best', 'Top3 Median (filter 150px)'),
        ('ensemble_filtered_median', 'Top3 Median (filtered)'),
        ('ensemble_filtered_mean', 'Top3 Mean (filtered)'),
        ('ensemble_filtered_trimmed20', 'Top3 Trimmed Mean 20%'),
        ('ensemble_filtered_trimmed10', 'Top3 Trimmed Mean 10%'),
        ('ensemble_all_median', 'All7 Median'),
        ('ensemble_all_weighted', 'All7 Weighted'),
        ('ensemble_all_consensus50', 'All7 Consensus 50px'),
        ('ensemble_all_consensus30', 'All7 Consensus 30px'),
        ('ensemble_all_consensus70', 'All7 Consensus 70px'),
        ('ensemble_top3_median', 'All7 Top3-Median'),
        ('ensemble_median_150', 'Top3 Median f150'),
        ('ensemble_mean_150', 'Top3 Mean f150'),
        ('ensemble_geom_median_150', 'Top3 Geom-Median f150'),
        ('ensemble_weighted_150', 'Top3 Weighted f150'),
        ('ensemble_best_only_150', 'Top3 Best-Only f150'),
        ('ensemble_two_closest_150', 'Top3 Two-Closest f150'),
        ('ensemble_median_100', 'Top3 Median f100'),
        ('ensemble_median_200', 'Top3 Median f200'),
        ('ensemble_median_250', 'Top3 Median f250'),
        ('ensemble_newtop3_median_150', 'NewTop3 Median f150'),
        ('ensemble_newtop3_mean_150', 'NewTop3 Mean f150'),
        ('ensemble_newtop3_weighted_150', 'NewTop3 Weighted f150'),
        ('ensemble_newtop3_median_100', 'NewTop3 Median f100'),
        ('ensemble_newtop3_median_200', 'NewTop3 Median f200'),
    ]

    print("\nEvaluating ensemble methods:")
    for file_suffix, display_name in ensemble_files:
        pred_path = root / f"train_predictions_{file_suffix}.csv"
        if pred_path.exists():
            pred = load_csv(pred_path)
            result = evaluate_predictions(pred, gt)
            if result:
                score, accuracies, stats = result
                results.append({
                    'name': display_name,
                    'type': 'ensemble',
                    'score': score,
                    'acc_5m': accuracies['5m'],
                    'acc_25m': accuracies['25m'],
                    'acc_100m': accuracies['100m'],
                    'median': stats['median'],
                    'mean': stats['mean'],
                    'n_samples': stats['n_samples']
                })
                print(f"  ✓ {display_name}")

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    # Display results
    print("\n" + "="*90)
    print("COMPLETE RANKING - ALL METHODS")
    print("="*90)

    print(f"\n{'Rank':<4} {'Method':<35} {'Type':<10} {'Score':>7} {'@5m':>7} {'@25m':>7} {'@100m':>7} {'Med':>6}")
    print("-" * 90)

    for i, r in enumerate(results, 1):
        marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker} {i:2d}  {r['name']:<35} {r['type']:<10} {r['score']:7.2f} "
              f"{r['acc_5m']:6.2f}% {r['acc_25m']:6.2f}% {r['acc_100m']:6.2f}% {r['median']:5.1f}px")

    # Summary statistics
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)

    best_ensemble = next((r for r in results if r['type'] == 'ensemble'), None)
    best_individual = next((r for r in results if r['type'] == 'individual'), None)

    if best_ensemble and best_individual:
        improvement = best_ensemble['score'] - best_individual['score']
        print(f"\n🏆 Best Overall:     {results[0]['name']:<35} Score: {results[0]['score']:.2f}")
        print(f"🥇 Best Ensemble:    {best_ensemble['name']:<35} Score: {best_ensemble['score']:.2f}")
        print(f"🥇 Best Individual:  {best_individual['name']:<35} Score: {best_individual['score']:.2f}")
        print(f"\n📊 Ensemble Improvement: +{improvement:.2f} points ({improvement/best_individual['score']*100:.1f}%)")

    # Top 5
    print(f"\n" + "="*90)
    print("TOP 5 METHODS")
    print("="*90)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['name']:<40} {r['score']:.2f} ({r['acc_5m']:.1f}% @ 5m)")

    # Count by type
    ensemble_count = sum(1 for r in results if r['type'] == 'ensemble')
    individual_count = sum(1 for r in results if r['type'] == 'individual')
    print(f"\n📈 Evaluated {len(results)} methods total: {ensemble_count} ensembles, {individual_count} individual models")

    print("="*90)


if __name__ == "__main__":
    main()
