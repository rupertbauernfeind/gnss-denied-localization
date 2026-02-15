#!/usr/bin/env python3
"""
Auto-discover all matchers from NPZ files and compare their performance.

Usage:
    python compare_all_methods.py
"""

import subprocess
import csv
import math
from pathlib import Path
from collections import defaultdict


def load_csv(path):
    """Load CSV into dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def evaluate_predictions(pred_path, gt_path):
    """Evaluate predictions and return score and accuracy breakdown."""
    gt = load_csv(gt_path)
    pred = load_csv(pred_path)

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
        'max': distances[-1],
        'n_samples': len(common_ids)
    }

    return score, accuracies, stats


def discover_matchers(sample_dir):
    """Discover all matchers from NPZ files in sample directory."""
    sample_path = Path(sample_dir)
    if not sample_path.exists():
        return []

    matchers = []
    for npz_file in sample_path.glob("*_matches.npz"):
        # Extract matcher name (everything before "_matches.npz")
        matcher_name = npz_file.stem.replace("_matches", "")
        matchers.append(matcher_name)

    return sorted(matchers)


def run_batch_analyze(matcher_name, root_dir):
    """Run batch_analyze.py for one matcher."""
    script_path = root_dir / "batch_analyze.py"

    # Activate conda and run
    cmd = f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate my && python {script_path} --refine --matcher={matcher_name}"

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(root_dir),
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )

    return result.returncode == 0, result.stdout, result.stderr


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    # Discover matchers from a sample image directory
    sample_dir = root / "match_visualizations" / "img_2836"
    print("="*80)
    print("AUTO-DISCOVERING MATCHERS FROM MATCH FILES")
    print("="*80)
    print(f"Sample directory: {sample_dir}\n")

    matchers = discover_matchers(sample_dir)

    if not matchers:
        print(f"✗ No matchers found in {sample_dir}")
        print("  Expected files like: <matcher>_matches.npz")
        return

    print(f"Found {len(matchers)} matchers:")
    for m in matchers:
        print(f"  - {m}")

    print("\n" + "="*80)
    print("RUNNING BATCH ANALYZE FOR ALL MATCHERS")
    print("="*80)

    # Run batch_analyze for each matcher
    results = {}
    for matcher in matchers:
        print(f"\n[{matcher}] Running batch_analyze.py...")
        success, stdout, stderr = run_batch_analyze(matcher, root)

        if success:
            print(f"  ✓ Generated predictions")
            results[matcher] = True
        else:
            print(f"  ✗ Failed to generate predictions")
            if stderr:
                print(f"  Error: {stderr[:200]}")
            results[matcher] = False

    # Evaluate all successful predictions
    print("\n" + "="*80)
    print("EVALUATING ALL METHODS")
    print("="*80)

    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"

    evaluations = []
    for matcher in matchers:
        if not results[matcher]:
            continue

        pred_path = root / f"train_predictions_{matcher}.csv"
        if not pred_path.exists():
            print(f"\n[{matcher}] ✗ Predictions file not found: {pred_path}")
            continue

        result = evaluate_predictions(pred_path, gt_path)
        if result is None:
            print(f"\n[{matcher}] ✗ No common samples to evaluate")
            continue

        score, accuracies, stats = result
        evaluations.append({
            'matcher': matcher,
            'score': score,
            'acc_5m': accuracies['5m'],
            'acc_25m': accuracies['25m'],
            'acc_100m': accuracies['100m'],
            'median': stats['median'],
            'mean': stats['mean'],
            'n_samples': stats['n_samples']
        })

        print(f"\n[{matcher}]")
        print(f"  Score: {score:.2f}")
        print(f"  Accuracy @ 5m:   {accuracies['5m']:6.2f}%")
        print(f"  Accuracy @ 25m:  {accuracies['25m']:6.2f}%")
        print(f"  Accuracy @ 100m: {accuracies['100m']:6.2f}%")
        print(f"  Median error: {stats['median']:.1f} px")
        print(f"  Mean error:   {stats['mean']:.1f} px")

    # Summary table
    if evaluations:
        print("\n" + "="*80)
        print("SUMMARY - SORTED BY SCORE")
        print("="*80)

        # Sort by score (descending)
        evaluations.sort(key=lambda x: x['score'], reverse=True)

        # Print header
        print(f"\n{'Matcher':<25} {'Score':>7} {'@5m':>7} {'@25m':>7} {'@100m':>7} {'Median':>8} {'Mean':>8}")
        print("-" * 80)

        # Print each matcher
        for ev in evaluations:
            print(f"{ev['matcher']:<25} {ev['score']:7.2f} {ev['acc_5m']:6.2f}% {ev['acc_25m']:6.2f}% {ev['acc_100m']:6.2f}% {ev['median']:7.1f}px {ev['mean']:7.1f}px")

        # Highlight best
        best = evaluations[0]
        print("\n" + "="*80)
        print(f"🏆 BEST METHOD: {best['matcher']} (score: {best['score']:.2f})")
        print("="*80)
    else:
        print("\n✗ No successful evaluations")

    print(f"\nProcessed {len(matchers)} matchers total")


if __name__ == "__main__":
    main()
