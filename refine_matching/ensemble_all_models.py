#!/usr/bin/env python3
"""
Ensemble ALL available models with advanced strategies.

Usage:
    python ensemble_all_models.py
"""

import csv
import math
import numpy as np
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


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def filter_valid_predictions(predictions, rough_pred, img_id, max_deviation=150.0):
    """Filter out failed predictions for a specific image."""
    if img_id not in rough_pred:
        return []

    rough_x, rough_y = rough_pred[img_id]
    valid = []

    for model_name, model_preds in predictions.items():
        if img_id not in model_preds:
            continue

        pred_x, pred_y = model_preds[img_id]

        # Skip if exactly equal to rough (failed to refine)
        if abs(pred_x - rough_x) < 0.01 and abs(pred_y - rough_y) < 0.01:
            continue

        # Skip if too far from rough (likely failure)
        dist = distance((pred_x, pred_y), (rough_x, rough_y))
        if dist > max_deviation:
            continue

        valid.append((pred_x, pred_y))

    return valid


def ensemble_median(valid_preds):
    """Median of all valid predictions."""
    if not valid_preds:
        return None
    x_coords = [p[0] for p in valid_preds]
    y_coords = [p[1] for p in valid_preds]
    return (np.median(x_coords), np.median(y_coords))


def ensemble_consensus_median(valid_preds, max_spread=50.0):
    """
    Only use predictions that are within max_spread of each other.
    If consensus found, return median. Otherwise return None.
    """
    if len(valid_preds) < 2:
        return ensemble_median(valid_preds) if valid_preds else None

    # Calculate centroid
    x_coords = [p[0] for p in valid_preds]
    y_coords = [p[1] for p in valid_preds]
    cx, cy = np.mean(x_coords), np.mean(y_coords)

    # Filter predictions close to centroid
    consensus = [p for p in valid_preds if distance(p, (cx, cy)) <= max_spread]

    if len(consensus) >= 2:  # Need at least 2 in agreement
        return ensemble_median(consensus)

    return None


def ensemble_weighted_by_agreement(valid_preds):
    """
    Weight each prediction by how close it is to the others.
    Predictions closer to consensus get more weight.
    """
    if len(valid_preds) < 2:
        return ensemble_median(valid_preds) if valid_preds else None

    # Calculate pairwise distances
    n = len(valid_preds)
    weights = []

    for i, pred in enumerate(valid_preds):
        # Weight is inverse of average distance to other predictions
        avg_dist = sum(distance(pred, other) for j, other in enumerate(valid_preds) if i != j) / (n - 1)
        weight = 1.0 / (1.0 + avg_dist)  # Inverse distance weight
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted average
    x_weighted = sum(p[0] * w for p, w in zip(valid_preds, weights))
    y_weighted = sum(p[1] * w for p, w in zip(valid_preds, weights))

    return (x_weighted, y_weighted)


def create_ensemble(predictions, rough_pred, ensemble_func):
    """Create ensemble predictions using specified aggregation function."""
    ensemble_pred = {}

    all_ids = set()
    for model_preds in predictions.values():
        all_ids.update(model_preds.keys())

    stats = {'used': 0, 'fallback': 0, 'total': 0}

    for img_id in sorted(all_ids):
        stats['total'] += 1

        valid_preds = filter_valid_predictions(predictions, rough_pred, img_id)
        result = ensemble_func(valid_preds)

        if result:
            ensemble_pred[img_id] = result
            stats['used'] += 1
        elif img_id in rough_pred:
            ensemble_pred[img_id] = rough_pred[img_id]
            stats['fallback'] += 1

    return ensemble_pred, stats


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    # Discover all available models
    models = ['xoftr', 'edm', 'superglue', 'rdd-lightglue', 'rdd-star', 'dedode', 'affine-steerers']

    print("="*80)
    print("ENSEMBLE WITH ALL MODELS")
    print("="*80)
    print(f"Models: {', '.join(models)}\n")

    # Load rough predictions
    rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    rough_pred = load_csv(rough_path)
    print(f"✓ Loaded rough predictions: {len(rough_pred)} images")

    # Load predictions from all models
    predictions = {}
    for model in models:
        pred_path = root / f"train_predictions_{model}.csv"
        if pred_path.exists():
            predictions[model] = load_csv(pred_path)
            print(f"✓ Loaded {model}: {len(predictions[model])} predictions")
        else:
            print(f"  Skipping {model} (no predictions file)")

    # Load ground truth
    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    gt = load_csv(gt_path)

    # Define ensemble strategies
    strategies = [
        ('all_median', 'All Models - Median', ensemble_median),
        ('all_weighted', 'All Models - Weighted by Agreement', ensemble_weighted_by_agreement),
        ('all_consensus50', 'All Models - Consensus (50px)', lambda p: ensemble_consensus_median(p, 50.0)),
        ('all_consensus30', 'All Models - Consensus (30px)', lambda p: ensemble_consensus_median(p, 30.0)),
        ('all_consensus70', 'All Models - Consensus (70px)', lambda p: ensemble_consensus_median(p, 70.0)),
    ]

    # Also try top 3 only for comparison
    top3_predictions = {k: v for k, v in predictions.items() if k in ['xoftr', 'edm', 'superglue']}
    strategies.append(('top3_median', 'Top 3 - Median',
                      lambda p: create_ensemble(top3_predictions, rough_pred, ensemble_median)[0]))

    print("\n" + "="*80)
    print("CREATING ENSEMBLES")
    print("="*80)

    results = []

    for strategy_key, strategy_name, ensemble_func in strategies:
        # Special handling for top3 comparison
        if strategy_key == 'top3_median':
            ensemble_pred, stats = create_ensemble(top3_predictions, rough_pred, ensemble_median)
        else:
            ensemble_pred, stats = create_ensemble(predictions, rough_pred, ensemble_func)

        # Save predictions
        output_path = root / f"train_predictions_ensemble_{strategy_key}.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
            writer.writeheader()
            for img_id in sorted(ensemble_pred.keys()):
                x, y = ensemble_pred[img_id]
                writer.writerow({'id': img_id, 'x_pixel': x, 'y_pixel': y})

        # Evaluate
        result = evaluate_predictions(ensemble_pred, gt)
        if result:
            score, accuracies, eval_stats = result
            results.append({
                'name': strategy_name,
                'key': strategy_key,
                'score': score,
                'acc_5m': accuracies['5m'],
                'acc_25m': accuracies['25m'],
                'acc_100m': accuracies['100m'],
                'median': eval_stats['median'],
                'mean': eval_stats['mean'],
                'n_samples': eval_stats['n_samples'],
                'n_ensembled': stats['used'],
                'n_fallback': stats['fallback']
            })

            print(f"\n{strategy_name}")
            print(f"  Ensembled: {stats['used']}/{stats['total']}, fallback: {stats['fallback']}")
            print(f"  Score: {score:.2f}  @5m: {accuracies['5m']:.2f}%  Median: {eval_stats['median']:.1f}px")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY - SORTED BY SCORE")
    print("="*80)

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'Method':<35} {'Score':>7} {'@5m':>7} {'Median':>8} {'Ens':>4} {'Fall':>4}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<35} {r['score']:7.2f} {r['acc_5m']:6.2f}% {r['median']:7.1f}px "
              f"{r['n_ensembled']:4d} {r['n_fallback']:4d}")

    best = results[0]

    print("\n" + "="*80)
    print(f"🏆 BEST: {best['name']}")
    print(f"   Score: {best['score']:.2f}  @5m: {best['acc_5m']:.2f}%  Median: {best['median']:.1f}px")
    print(f"   Ensembled: {best['n_ensembled']}, Fallbacks: {best['n_fallback']}")
    print("="*80)


if __name__ == "__main__":
    main()
