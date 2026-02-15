#!/usr/bin/env python3
"""
Advanced ensemble methods with failure detection and multiple aggregation strategies.

Usage:
    python ensemble_advanced.py
"""

import csv
import math
import numpy as np
from pathlib import Path
from collections import defaultdict


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
    """
    Filter out failed predictions for a specific image.

    Returns list of valid (x, y) tuples.
    Excludes predictions that:
    - Are exactly equal to rough guess (method failed)
    - Are more than max_deviation pixels from rough guess (likely wrong)
    """
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


def ensemble_mean(valid_preds):
    """Mean of all valid predictions."""
    if not valid_preds:
        return None
    x_coords = [p[0] for p in valid_preds]
    y_coords = [p[1] for p in valid_preds]
    return (np.mean(x_coords), np.mean(y_coords))


def ensemble_trimmed_mean(valid_preds, trim_fraction=0.2):
    """Trimmed mean - remove top/bottom outliers before averaging."""
    if not valid_preds or len(valid_preds) < 3:
        return ensemble_mean(valid_preds) if valid_preds else None

    x_coords = [p[0] for p in valid_preds]
    y_coords = [p[1] for p in valid_preds]

    from scipy import stats
    x_trimmed = stats.trim_mean(x_coords, trim_fraction)
    y_trimmed = stats.trim_mean(y_coords, trim_fraction)

    return (x_trimmed, y_trimmed)


def ensemble_weighted_mean(valid_preds, model_scores):
    """Weighted mean using model scores as weights."""
    if not valid_preds:
        return None

    # For now, just use mean (we'd need to track which pred came from which model)
    return ensemble_mean(valid_preds)


def ensemble_best_model_per_image(predictions, rough_pred, model_names, img_id, gt):
    """
    For each image, pick the prediction from the model that's closest to ground truth.
    (This is cheating for evaluation but shows upper bound)
    """
    if img_id not in gt:
        return None

    gt_pos = gt[img_id]
    best_pred = None
    best_dist = float('inf')

    for model in model_names:
        if img_id not in predictions[model]:
            continue
        pred_pos = predictions[model][img_id]
        dist = distance(pred_pos, gt_pos)
        if dist < best_dist:
            best_dist = dist
            best_pred = pred_pos

    return best_pred


def create_ensemble(predictions, rough_pred, method_name, ensemble_func, model_names=None, gt=None):
    """Create ensemble predictions using specified aggregation function."""
    ensemble_pred = {}

    # Get all image IDs that have at least one prediction
    all_ids = set()
    for model_preds in predictions.values():
        all_ids.update(model_preds.keys())

    stats = {'used': 0, 'fallback': 0, 'total': 0}

    for img_id in sorted(all_ids):
        stats['total'] += 1

        # Special case for best-model-per-image
        if method_name == 'best_per_image':
            result = ensemble_best_model_per_image(predictions, rough_pred, model_names, img_id, gt)
            if result:
                ensemble_pred[img_id] = result
                stats['used'] += 1
            continue

        # Filter valid predictions for this image
        valid_preds = filter_valid_predictions(predictions, rough_pred, img_id)

        # Apply ensemble function
        result = ensemble_func(valid_preds)

        if result:
            ensemble_pred[img_id] = result
            stats['used'] += 1
        elif img_id in rough_pred:
            # Fallback to rough prediction if all methods failed
            ensemble_pred[img_id] = rough_pred[img_id]
            stats['fallback'] += 1

    return ensemble_pred, stats


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    # Top models to ensemble
    models = ['xoftr', 'edm', 'superglue']

    print("="*80)
    print("ADVANCED ENSEMBLE COMPARISON")
    print("="*80)
    print(f"Models: {', '.join(models)}")
    print("Filtering: >150px from rough OR unchanged from rough\n")

    # Load rough predictions
    rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    rough_pred = load_csv(rough_path)
    print(f"✓ Loaded rough predictions: {len(rough_pred)} images")

    # Load predictions from all models
    predictions = {}
    for model in models:
        pred_path = root / f"train_predictions_{model}.csv"
        if not pred_path.exists():
            print(f"✗ Missing: {pred_path}")
            return
        predictions[model] = load_csv(pred_path)
        print(f"✓ Loaded {model}: {len(predictions[model])} predictions")

    # Load ground truth
    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    gt = load_csv(gt_path)

    # Define ensemble strategies
    strategies = [
        ('filtered_median', 'Median (filtered)', ensemble_median),
        ('filtered_mean', 'Mean (filtered)', ensemble_mean),
        ('filtered_trimmed20', 'Trimmed Mean 20% (filtered)',
         lambda preds: ensemble_trimmed_mean(preds, 0.2)),
        ('filtered_trimmed10', 'Trimmed Mean 10% (filtered)',
         lambda preds: ensemble_trimmed_mean(preds, 0.1)),
    ]

    print("\n" + "="*80)
    print("CREATING ENSEMBLES")
    print("="*80)

    results = []

    for strategy_key, strategy_name, ensemble_func in strategies:
        ensemble_pred, stats = create_ensemble(
            predictions, rough_pred, strategy_key, ensemble_func
        )

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
            print(f"  Images ensembled: {stats['used']}/{stats['total']}, fallback: {stats['fallback']}")
            print(f"  Score: {score:.2f}  @5m: {accuracies['5m']:.2f}%  Median: {eval_stats['median']:.1f}px")

    # Also evaluate individual models for comparison
    print("\n" + "="*80)
    print("INDIVIDUAL MODELS (for reference)")
    print("="*80)

    individual_results = []
    for model in models:
        result = evaluate_predictions(predictions[model], gt)
        if result:
            score, accuracies, stats = result
            individual_results.append({
                'name': model,
                'score': score,
                'acc_5m': accuracies['5m']
            })
            print(f"{model:20} - Score: {score:.2f}  @5m: {accuracies['5m']:6.2f}%")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY - SORTED BY SCORE")
    print("="*80)

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'Method':<30} {'Score':>7} {'@5m':>7} {'@25m':>7} {'@100m':>7} {'Median':>8} {'Mean':>8} {'N':>5}")
    print("-" * 95)

    for r in results:
        print(f"{r['name']:<30} {r['score']:7.2f} {r['acc_5m']:6.2f}% {r['acc_25m']:6.2f}% "
              f"{r['acc_100m']:6.2f}% {r['median']:7.1f}px {r['mean']:7.1f}px {r['n_ensembled']:5d}")

    best = results[0]
    best_individual = max(individual_results, key=lambda x: x['score'])

    print("\n" + "="*80)
    print(f"🏆 BEST ENSEMBLE: {best['name']}")
    print(f"   Score: {best['score']:.2f} (+{best['score'] - best_individual['score']:.2f} over best individual)")
    print(f"   @5m: {best['acc_5m']:.2f}%")
    print(f"   Ensembled {best['n_ensembled']} images, {best['n_fallback']} fallbacks")
    print("="*80)


if __name__ == "__main__":
    main()
