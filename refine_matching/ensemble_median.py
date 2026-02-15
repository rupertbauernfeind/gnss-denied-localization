#!/usr/bin/env python3
"""
Ensemble top models by taking median of their predictions.

Usage:
    python ensemble_median.py
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


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    # Top 3 models
    models = ['xoftr', 'edm', 'superglue']

    print("="*80)
    print("ENSEMBLE: MEDIAN OF TOP 3 MODELS")
    print("="*80)
    print(f"Models: {', '.join(models)}\n")

    # Load predictions from all models
    predictions = {}
    for model in models:
        pred_path = root / f"train_predictions_{model}.csv"
        if not pred_path.exists():
            print(f"✗ Missing: {pred_path}")
            return
        predictions[model] = load_csv(pred_path)
        print(f"✓ Loaded {model}: {len(predictions[model])} predictions")

    # Find common IDs across all models
    common_ids = set(predictions[models[0]].keys())
    for model in models[1:]:
        common_ids &= set(predictions[model].keys())

    common_ids = sorted(common_ids)
    print(f"\nCommon predictions across all models: {len(common_ids)}")

    # Calculate median coordinates
    ensemble_pred = {}
    for img_id in common_ids:
        x_coords = [predictions[model][img_id][0] for model in models]
        y_coords = [predictions[model][img_id][1] for model in models]

        median_x = np.median(x_coords)
        median_y = np.median(y_coords)

        ensemble_pred[img_id] = (median_x, median_y)

    # Save ensemble predictions
    output_path = root / "train_predictions_ensemble_median.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        for img_id in sorted(ensemble_pred.keys()):
            x, y = ensemble_pred[img_id]
            writer.writerow({'id': img_id, 'x_pixel': x, 'y_pixel': y})

    print(f"✓ Saved ensemble predictions to {output_path.name}")

    # Evaluate ensemble
    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    gt = load_csv(gt_path)

    result = evaluate_predictions(ensemble_pred, gt)
    if result is None:
        print("✗ No common samples to evaluate")
        return

    score, accuracies, stats = result

    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(f"  Score: {score:.2f}")
    print(f"  Accuracy @ 5m:   {accuracies['5m']:6.2f}%")
    print(f"  Accuracy @ 25m:  {accuracies['25m']:6.2f}%")
    print(f"  Accuracy @ 100m: {accuracies['100m']:6.2f}%")
    print(f"  Median error: {stats['median']:.1f} px")
    print(f"  Mean error:   {stats['mean']:.1f} px")

    # Compare to individual models
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    individual_scores = []
    for model in models:
        result = evaluate_predictions(predictions[model], gt)
        if result:
            model_score, model_acc, model_stats = result
            individual_scores.append(model_score)
            print(f"{model:12} - Score: {model_score:.2f}  @5m: {model_acc['5m']:6.2f}%")

    print(f"{'ensemble':12} - Score: {score:.2f}  @5m: {accuracies['5m']:6.2f}%")

    best_individual = max(individual_scores)
    improvement = score - best_individual

    print("\n" + "="*80)
    if improvement > 0:
        print(f"🎉 Ensemble IMPROVES by +{improvement:.2f} over best individual!")
    elif improvement < 0:
        print(f"⚠️  Ensemble is {-improvement:.2f} worse than best individual")
    else:
        print("Ensemble matches best individual")
    print("="*80)


if __name__ == "__main__":
    main()
