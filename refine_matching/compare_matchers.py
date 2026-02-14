#!/usr/bin/env python3
"""
Compare multiple matcher results side-by-side.

Usage:
    python compare_matchers.py master se2loftr
"""

import csv
import math
import sys
from pathlib import Path


def load_csv(path):
    """Load CSV into dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def evaluate_predictions(pred, gt, name):
    """Evaluate predictions and return statistics."""
    common_ids = sorted(set(gt) & set(pred))

    if len(common_ids) == 0:
        return None

    distances = []
    for i in common_ids:
        dx = pred[i][0] - gt[i][0]
        dy = pred[i][1] - gt[i][1]
        distances.append(math.sqrt(dx * dx + dy * dy))

    thresholds = [(25, "5m"), (125, "25m"), (500, "100m")]
    accuracies = []
    for px_thresh, label in thresholds:
        acc = sum(1 for d in distances if d <= px_thresh) / len(distances) * 100
        accuracies.append(acc)

    score = sum(accuracies) / len(accuracies)

    distances.sort()
    median = distances[len(distances)//2]
    mean = sum(distances)/len(distances)

    return {
        'name': name,
        'n_samples': len(common_ids),
        'acc_5m': accuracies[0],
        'acc_25m': accuracies[1],
        'acc_100m': accuracies[2],
        'score': score,
        'median': median,
        'mean': mean,
        'max': distances[-1]
    }


def main():
    """Compare multiple matchers."""
    root = Path(__file__).parent
    repo_root = root.parent
    gt = load_csv(repo_root / "data" / "train_data" / "train_pos.csv")

    # Get matchers from command line or use defaults
    if len(sys.argv) > 1:
        matchers = sys.argv[1:]
    else:
        matchers = ['master', 'se2loftr']

    print("="*100)
    print("MATCHER COMPARISON")
    print("="*100)

    results = []

    for matcher in matchers:
        # Determine file name (master has no suffix, others do)
        suffix = f"_{matcher}" if matcher != 'master' else ""
        pred_file = root / f"train_predictions{suffix}.csv"

        if not pred_file.exists():
            print(f"\n⚠ Skipping {matcher}: {pred_file} not found")
            continue

        pred = load_csv(pred_file)
        stats = evaluate_predictions(pred, gt, matcher)

        if stats:
            results.append(stats)

    if not results:
        print("\n✗ No valid prediction files found!")
        print("\nExpected files:")
        for matcher in matchers:
            suffix = f"_{matcher}" if matcher != 'master' else ""
            print(f"  - train_predictions{suffix}.csv")
        return

    # Display results
    print(f"\n{'Matcher':<15} {'Score':>8} {'@5m':>8} {'@25m':>8} {'@100m':>8} {'Median':>10} {'Mean':>10} {'Samples':>8}")
    print("-"*100)

    for stats in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{stats['name']:<15} "
              f"{stats['score']:>8.2f} "
              f"{stats['acc_5m']:>7.1f}% "
              f"{stats['acc_25m']:>7.1f}% "
              f"{stats['acc_100m']:>7.1f}% "
              f"{stats['median']:>9.1f}px "
              f"{stats['mean']:>9.1f}px "
              f"{stats['n_samples']:>8}")

    # Comparison
    if len(results) >= 2:
        best = max(results, key=lambda x: x['score'])
        worst = min(results, key=lambda x: x['score'])

        print("\n" + "="*100)
        print("COMPARISON")
        print("="*100)
        print(f"\nBest matcher: {best['name']}")
        print(f"  Score: {best['score']:.2f}")
        print(f"  Median error: {best['median']:.1f}px")

        if best['name'] != worst['name']:
            score_diff = best['score'] - worst['score']
            median_diff = worst['median'] - best['median']
            print(f"\nImprovement over {worst['name']}:")
            print(f"  Score: +{score_diff:.2f} points ({score_diff/worst['score']*100:.1f}%)")
            print(f"  Median: -{median_diff:.1f}px ({median_diff/worst['median']*100:.1f}% reduction)")

    print("="*100)


if __name__ == "__main__":
    main()
