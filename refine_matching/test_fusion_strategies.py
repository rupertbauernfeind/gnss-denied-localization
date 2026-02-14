#!/usr/bin/env python3
"""
Test different strategies for combining refined predictions with rough GPS.

Instead of just using a hard cutoff (e.g., 140px), this script tests various
fusion approaches to find the optimal way to combine the two position estimates.
"""

import csv
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_csv(path):
    """Load CSV file into dict mapping image_id -> (x, y)."""
    result = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['id'])
            x = row['x_pixel']
            y = row['y_pixel']

            # Handle None values (failed refinements)
            if x is None or y is None or x == '' or y == '':
                result[img_id] = None
            else:
                result[img_id] = (float(x), float(y))
    return result


def calculate_competition_score(errors):
    """Calculate competition score: mean of accuracies at 5m, 25m, 100m."""
    if len(errors) == 0:
        return 0.0

    acc_5m = sum(1 for e in errors if e <= 25) / len(errors) * 100
    acc_25m = sum(1 for e in errors if e <= 125) / len(errors) * 100
    acc_100m = sum(1 for e in errors if e <= 500) / len(errors) * 100

    return (acc_5m + acc_25m + acc_100m) / 3


def fusion_hard_cutoff(rough_pos, refined_pos, threshold):
    """
    Hard cutoff: use refined if within threshold, otherwise use rough.

    Args:
        rough_pos: (x, y) rough GPS position
        refined_pos: (x, y) refined position or None if failed
        threshold: maximum allowed movement in pixels

    Returns:
        (x, y) fused position
    """
    if refined_pos is None:
        return rough_pos

    rough_x, rough_y = rough_pos
    refined_x, refined_y = refined_pos

    movement = np.sqrt((refined_x - rough_x)**2 + (refined_y - rough_y)**2)

    if movement <= threshold:
        return refined_pos
    else:
        return rough_pos


def fusion_weighted_average(rough_pos, refined_pos, sigma):
    """
    Weighted average based on distance from rough GPS.

    Weight function: w = exp(-distance^2 / (2*sigma^2))
    fused = w * refined + (1-w) * rough

    Args:
        rough_pos: (x, y) rough GPS position
        refined_pos: (x, y) refined position or None if failed
        sigma: controls how quickly we transition from refined to rough

    Returns:
        (x, y) fused position
    """
    if refined_pos is None:
        return rough_pos

    rough_x, rough_y = rough_pos
    refined_x, refined_y = refined_pos

    distance = np.sqrt((refined_x - rough_x)**2 + (refined_y - rough_y)**2)

    # Gaussian weight - closer to rough GPS gets higher weight for refined
    weight_refined = np.exp(-distance**2 / (2 * sigma**2))
    weight_rough = 1 - weight_refined

    fused_x = weight_refined * refined_x + weight_rough * rough_x
    fused_y = weight_refined * refined_y + weight_rough * rough_y

    return fused_x, fused_y


def fusion_linear_blend(rough_pos, refined_pos, d_min, d_max):
    """
    Linear blending based on distance.

    If distance < d_min: use refined 100%
    If distance > d_max: use rough 100%
    In between: linear interpolation

    Args:
        rough_pos: (x, y) rough GPS position
        refined_pos: (x, y) refined position or None if failed
        d_min: distance below which we trust refined 100%
        d_max: distance above which we use rough 100%

    Returns:
        (x, y) fused position
    """
    if refined_pos is None:
        return rough_pos

    rough_x, rough_y = rough_pos
    refined_x, refined_y = refined_pos

    distance = np.sqrt((refined_x - rough_x)**2 + (refined_y - rough_y)**2)

    if distance <= d_min:
        # Close enough - trust refined completely
        return refined_pos
    elif distance >= d_max:
        # Too far - use rough completely
        return rough_pos
    else:
        # Linear blend
        alpha = (distance - d_min) / (d_max - d_min)  # 0 to 1
        fused_x = (1 - alpha) * refined_x + alpha * rough_x
        fused_y = (1 - alpha) * refined_y + alpha * rough_y
        return fused_x, fused_y


def fusion_always_refined(rough_pos, refined_pos):
    """Always use refined (if available), otherwise rough."""
    if refined_pos is None:
        return rough_pos
    return refined_pos


def fusion_always_rough(rough_pos, refined_pos):
    """Always use rough GPS (baseline)."""
    return rough_pos


def fusion_simple_average(rough_pos, refined_pos):
    """Simple 50/50 average of rough and refined."""
    if refined_pos is None:
        return rough_pos

    rough_x, rough_y = rough_pos
    refined_x, refined_y = refined_pos

    return (rough_x + refined_x) / 2, (rough_y + refined_y) / 2


def evaluate_fusion_strategy(fusion_func, rough_dict, refined_dict, gt_dict, strategy_name):
    """
    Evaluate a fusion strategy on train images with ground truth.

    Returns:
        dict with statistics about this fusion strategy
    """
    errors = []

    for img_id in sorted(gt_dict.keys()):
        if img_id not in rough_dict:
            continue

        gt_x, gt_y = gt_dict[img_id]
        rough_pos = rough_dict[img_id]
        refined_pos = refined_dict.get(img_id, None)

        # Apply fusion strategy
        fused_x, fused_y = fusion_func(rough_pos, refined_pos)

        # Calculate error
        error = np.sqrt((fused_x - gt_x)**2 + (fused_y - gt_y)**2)
        errors.append(error)

    if len(errors) == 0:
        return None

    # Calculate statistics
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)

    # Competition score
    score = calculate_competition_score(errors)

    acc_5m = sum(1 for e in errors if e <= 25) / len(errors) * 100
    acc_25m = sum(1 for e in errors if e <= 125) / len(errors) * 100
    acc_100m = sum(1 for e in errors if e <= 500) / len(errors) * 100

    return {
        'name': strategy_name,
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'score': score,
        'acc_5m': acc_5m,
        'acc_25m': acc_25m,
        'acc_100m': acc_100m,
        'n': len(errors),
    }


def main():
    """Test all fusion strategies and report results."""
    print("="*100)
    print("FUSION STRATEGY TESTING")
    print("Testing different ways to combine refined predictions with rough GPS")
    print("="*100)

    # Load data
    root = Path(__file__).parent.parent

    print("\nLoading data...")
    rough_dict = load_csv(root / "rough_matching" / "train_predictions.csv")
    refined_dict = load_csv(Path(__file__).parent / "train_refined_raw.csv")
    gt_dict = load_csv(root / "data" / "train_data" / "train_pos.csv")

    print(f"  Rough GPS predictions:   {len(rough_dict)}")
    print(f"  Refined predictions:     {len(refined_dict)}")
    print(f"  Ground truth:            {len(gt_dict)}")

    # Count how many refined predictions are available
    refined_available = sum(1 for v in refined_dict.values() if v is not None)
    print(f"  Refined successful:      {refined_available}/{len(refined_dict)} ({refined_available/len(refined_dict)*100:.1f}%)")

    # Define fusion strategies to test
    strategies = []

    # Baseline: always rough
    strategies.append(("Always Rough GPS (baseline)", lambda r, f: fusion_always_rough(r, f)))

    # Always refined (when available)
    strategies.append(("Always Refined (no validation)", lambda r, f: fusion_always_refined(r, f)))

    # Simple average
    strategies.append(("Simple 50/50 Average", lambda r, f: fusion_simple_average(r, f)))

    # Hard cutoff at various thresholds
    for threshold in [50, 75, 100, 125, 140, 150, 175, 200, 250]:
        strategies.append((
            f"Hard cutoff {threshold}px",
            lambda r, f, t=threshold: fusion_hard_cutoff(r, f, t)
        ))

    # Weighted average with different sigma values
    for sigma in [50, 75, 100, 125, 150, 200]:
        strategies.append((
            f"Weighted avg σ={sigma}",
            lambda r, f, s=sigma: fusion_weighted_average(r, f, s)
        ))

    # Linear blend with different ranges
    for d_min, d_max in [(25, 100), (50, 150), (50, 200), (75, 150), (100, 200)]:
        strategies.append((
            f"Linear blend [{d_min}-{d_max}]px",
            lambda r, f, dmin=d_min, dmax=d_max: fusion_linear_blend(r, f, dmin, dmax)
        ))

    # Evaluate all strategies
    print("\nEvaluating fusion strategies...")
    results = []

    for strategy_name, fusion_func in strategies:
        stats = evaluate_fusion_strategy(fusion_func, rough_dict, refined_dict, gt_dict, strategy_name)
        if stats is not None:
            results.append(stats)

    # Sort by competition score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print results
    print("\n" + "="*120)
    print("FUSION STRATEGIES RANKED BY COMPETITION SCORE")
    print("="*120)
    print(f"{'Rank':<6} {'Strategy':<35} {'Median':<10} {'Mean':<10} {'Score':<10} {'@5m':<8} {'@25m':<8} {'@100m':<8}")
    print("-"*120)

    for rank, stats in enumerate(results, 1):
        print(f"{rank:<6} {stats['name']:<35} "
              f"{stats['median']:>8.1f}px "
              f"{stats['mean']:>8.1f}px "
              f"{stats['score']:>8.2f}  "
              f"{stats['acc_5m']:>6.1f}% "
              f"{stats['acc_25m']:>6.1f}% "
              f"{stats['acc_100m']:>6.1f}%")

    # Summary
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)

    if results:
        best = results[0]
        baseline = next((r for r in results if r['name'] == "Always Rough GPS (baseline)"), None)

        print(f"\nBEST FUSION STRATEGY: {best['name']}")
        print(f"  Competition score: {best['score']:.2f}", end="")
        if baseline:
            print(f" (vs {baseline['score']:.2f} baseline, Δ={best['score']-baseline['score']:+.2f})")
        else:
            print()

        print(f"  Median error:      {best['median']:.1f}px", end="")
        if baseline:
            print(f" (vs {baseline['median']:.1f}px baseline, Δ={best['median']-baseline['median']:+.1f}px)")
        else:
            print()

        print(f"  Mean error:        {best['mean']:.1f}px", end="")
        if baseline:
            print(f" (vs {baseline['mean']:.1f}px baseline, Δ={best['mean']-baseline['mean']:+.1f}px)")
        else:
            print()

        print(f"\n  Accuracy breakdown:")
        print(f"    @5m (25px):    {best['acc_5m']:.1f}%", end="")
        if baseline:
            print(f" (vs {baseline['acc_5m']:.1f}%, Δ={best['acc_5m']-baseline['acc_5m']:+.1f}%)")
        else:
            print()

        print(f"    @25m (125px):  {best['acc_25m']:.1f}%", end="")
        if baseline:
            print(f" (vs {baseline['acc_25m']:.1f}%, Δ={best['acc_25m']-baseline['acc_25m']:+.1f}%)")
        else:
            print()

        print(f"    @100m (500px): {best['acc_100m']:.1f}%", end="")
        if baseline:
            print(f" (vs {baseline['acc_100m']:.1f}%, Δ={best['acc_100m']-baseline['acc_100m']:+.1f}%)")
        else:
            print()

        if baseline:
            improvement = best['score'] - baseline['score']
            if improvement > 0:
                print(f"\n✓ IMPROVEMENT: +{improvement:.2f} points ({improvement/baseline['score']*100:.1f}%)")
            elif improvement < 0:
                print(f"\n✗ REGRESSION: {improvement:.2f} points ({improvement/baseline['score']*100:.1f}%)")
            else:
                print(f"\n= NO CHANGE")

    print("="*120)


if __name__ == "__main__":
    main()
