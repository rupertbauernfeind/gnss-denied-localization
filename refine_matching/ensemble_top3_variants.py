#!/usr/bin/env python3
"""
Comprehensive ensemble testing on top 3 models (XoFTR, EDM, SuperGlue).

Usage:
    python ensemble_top3_variants.py
"""

import csv
import math
import numpy as np
from pathlib import Path
from scipy.spatial import distance as scipy_distance


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

        valid.append((pred_x, pred_y, model_name))

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


def ensemble_geometric_median(valid_preds):
    """Geometric median - point that minimizes sum of distances to all others."""
    if not valid_preds:
        return None
    if len(valid_preds) == 1:
        return valid_preds[0][:2]

    points = np.array([(p[0], p[1]) for p in valid_preds])

    # Use coordinate median as starting point
    current = np.median(points, axis=0)

    # Weiszfeld's algorithm
    for _ in range(100):
        distances = np.linalg.norm(points - current, axis=1)
        distances = np.where(distances == 0, 1e-10, distances)  # Avoid division by zero
        weights = 1.0 / distances
        new = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)

        if np.linalg.norm(new - current) < 1e-6:
            break
        current = new

    return tuple(current)


def ensemble_weighted_by_accuracy(valid_preds, model_scores):
    """Weighted average using model scores."""
    if not valid_preds:
        return None

    x_weighted = 0
    y_weighted = 0
    total_weight = 0

    for pred in valid_preds:
        x, y, model_name = pred[0], pred[1], pred[2]
        weight = model_scores.get(model_name, 1.0)
        x_weighted += x * weight
        y_weighted += y * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return (x_weighted / total_weight, y_weighted / total_weight)


def ensemble_best_only(valid_preds, model_scores):
    """Just use the prediction from the best model."""
    if not valid_preds:
        return None

    # Find prediction from model with highest score
    best_pred = None
    best_score = -1

    for pred in valid_preds:
        model_name = pred[2]
        score = model_scores.get(model_name, 0)
        if score > best_score:
            best_score = score
            best_pred = (pred[0], pred[1])

    return best_pred


def ensemble_two_closest(valid_preds):
    """Average the two predictions that are closest to each other."""
    if not valid_preds:
        return None
    if len(valid_preds) == 1:
        return (valid_preds[0][0], valid_preds[0][1])
    if len(valid_preds) == 2:
        return ensemble_mean(valid_preds)

    # Find two closest predictions
    min_dist = float('inf')
    best_pair = (0, 1)

    for i in range(len(valid_preds)):
        for j in range(i + 1, len(valid_preds)):
            d = distance((valid_preds[i][0], valid_preds[i][1]),
                        (valid_preds[j][0], valid_preds[j][1]))
            if d < min_dist:
                min_dist = d
                best_pair = (i, j)

    p1 = valid_preds[best_pair[0]]
    p2 = valid_preds[best_pair[1]]

    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def create_ensemble(predictions, rough_pred, ensemble_func, max_deviation=150.0,
                   model_scores=None):
    """Create ensemble predictions using specified aggregation function."""
    ensemble_pred = {}

    all_ids = set()
    for model_preds in predictions.values():
        all_ids.update(model_preds.keys())

    stats = {'used': 0, 'fallback': 0, 'total': 0}

    for img_id in sorted(all_ids):
        stats['total'] += 1

        valid_preds = filter_valid_predictions(predictions, rough_pred, img_id,
                                               max_deviation)

        # Pass model_scores if needed
        if model_scores and ensemble_func.__name__ in ['ensemble_weighted_by_accuracy',
                                                         'ensemble_best_only']:
            result = ensemble_func(valid_preds, model_scores)
        else:
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

    # Top 3 models
    models = ['xoftr', 'edm', 'superglue']

    # Model scores (from previous evaluation)
    model_scores = {
        'xoftr': 89.00,
        'edm': 88.71,
        'superglue': 86.19
    }

    print("="*80)
    print("TOP 3 ENSEMBLE VARIANTS")
    print("="*80)
    print(f"Models: {', '.join(models)}\n")

    # Load rough predictions
    rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    rough_pred = load_csv(rough_path)
    print(f"✓ Loaded rough predictions: {len(rough_pred)} images")

    # Load predictions from top 3 models
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
        # Different aggregation functions
        ('median_150', 'Median (filter 150px)', ensemble_median, 150),
        ('mean_150', 'Mean (filter 150px)', ensemble_mean, 150),
        ('geom_median_150', 'Geometric Median (filter 150px)', ensemble_geometric_median, 150),
        ('weighted_150', 'Weighted by Accuracy (filter 150px)', ensemble_weighted_by_accuracy, 150),
        ('best_only_150', 'Best Model Only (filter 150px)', ensemble_best_only, 150),
        ('two_closest_150', 'Two Closest Average (filter 150px)', ensemble_two_closest, 150),

        # Different filter thresholds with median
        ('median_100', 'Median (filter 100px)', ensemble_median, 100),
        ('median_200', 'Median (filter 200px)', ensemble_median, 200),
        ('median_250', 'Median (filter 250px)', ensemble_median, 250),

        # Different filter thresholds with mean
        ('mean_100', 'Mean (filter 100px)', ensemble_mean, 100),
        ('mean_200', 'Mean (filter 200px)', ensemble_mean, 200),

        # Weighted with different thresholds
        ('weighted_100', 'Weighted by Accuracy (filter 100px)', ensemble_weighted_by_accuracy, 100),
        ('weighted_200', 'Weighted by Accuracy (filter 200px)', ensemble_weighted_by_accuracy, 200),
    ]

    print("\n" + "="*80)
    print("CREATING ENSEMBLES")
    print("="*80)

    results = []

    for strategy_key, strategy_name, ensemble_func, max_dev in strategies:
        ensemble_pred, stats = create_ensemble(
            predictions, rough_pred, ensemble_func, max_dev, model_scores
        )

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

            print(f"{strategy_name:<45} Score: {score:5.2f}  @5m: {accuracies['5m']:5.2f}%  "
                  f"Ens: {stats['used']:3d}  Fall: {stats['fallback']:2d}")

    # Summary table
    print("\n" + "="*80)
    print("TOP 10 STRATEGIES - SORTED BY SCORE")
    print("="*80)

    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'Method':<45} {'Score':>7} {'@5m':>7} {'Median':>8} {'Ens':>4} {'Fall':>4}")
    print("-" * 90)

    for r in results[:10]:
        print(f"{r['name']:<45} {r['score']:7.2f} {r['acc_5m']:6.2f}% {r['median']:7.1f}px "
              f"{r['n_ensembled']:4d} {r['n_fallback']:4d}")

    best = results[0]

    print("\n" + "="*80)
    print(f"🏆 BEST: {best['name']}")
    print(f"   Score: {best['score']:.2f}  @5m: {best['acc_5m']:.2f}%  Median: {best['median']:.1f}px")
    print(f"   Ensembled: {best['n_ensembled']}, Fallbacks: {best['n_fallback']}")
    print("="*80)

    # Save best ensemble
    best_key = best['key']
    for strategy_key, strategy_name, ensemble_func, max_dev in strategies:
        if strategy_key == best_key:
            ensemble_pred, _ = create_ensemble(
                predictions, rough_pred, ensemble_func, max_dev, model_scores
            )

            output_path = root / f"train_predictions_ensemble_best.csv"
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
                writer.writeheader()
                for img_id in sorted(ensemble_pred.keys()):
                    x, y = ensemble_pred[img_id]
                    writer.writerow({'id': img_id, 'x_pixel': x, 'y_pixel': y})

            print(f"\n✓ Saved best ensemble to {output_path.name}")
            break


if __name__ == "__main__":
    main()
