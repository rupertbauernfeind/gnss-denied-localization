#!/usr/bin/env python3
"""
Ensemble combining: rough GPS positions, se2loftr, and minima-roma with -3px y-offset.

Usage:
    python ensemble_rough_se2_roma.py
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


def save_csv(path, data):
    """Save predictions to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        for img_id in sorted(data.keys()):
            x, y = data[img_id]
            writer.writerow({'id': img_id, 'x_pixel': x, 'y_pixel': y})


def apply_y_offset(predictions, offset):
    """Apply a constant offset to y-coordinates."""
    result = {}
    for img_id, (x, y) in predictions.items():
        result[img_id] = (x, y + offset)
    return result


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


def create_ensemble(predictions_dict):
    """
    Create ensemble by taking median of all available predictions.
    predictions_dict: {img_id: [(x1, y1), (x2, y2), ...]}
    """
    ensemble_pred = {}

    for img_id, preds in predictions_dict.items():
        if not preds:
            continue

        x_coords = [p[0] for p in preds]
        y_coords = [p[1] for p in preds]

        ensemble_pred[img_id] = (np.median(x_coords), np.median(y_coords))

    return ensemble_pred


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    print("="*80)
    print("ENSEMBLE: ROUGH + SE2LOFTR + MINIMA-ROMA(-3px)")
    print("="*80)

    # Load ground truth for evaluation
    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    gt = load_csv(gt_path)

    # ===== TRAIN PREDICTIONS =====
    print("\nTRAIN PREDICTIONS")
    print("-" * 80)

    # Load rough predictions
    train_rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    train_rough = load_csv(train_rough_path)
    print(f"✓ Loaded rough: {len(train_rough)} predictions")

    # Load se2loftr
    train_se2_path = root / "train_predictions_se2loftr.csv"
    train_se2 = load_csv(train_se2_path)
    print(f"✓ Loaded se2loftr: {len(train_se2)} predictions")

    # Load minima-roma and apply -3px offset
    train_roma_path = root / "train_predictions_minima-roma.csv"
    train_roma_orig = load_csv(train_roma_path)
    train_roma = apply_y_offset(train_roma_orig, -3)
    print(f"✓ Loaded minima-roma: {len(train_roma)} predictions (applied -3px y-offset)")

    # Combine all predictions per image
    all_train_ids = set(train_rough.keys()) | set(train_se2.keys()) | set(train_roma.keys())
    train_combined = {}

    for img_id in all_train_ids:
        preds = []
        if img_id in train_rough:
            preds.append(train_rough[img_id])
        if img_id in train_se2:
            preds.append(train_se2[img_id])
        if img_id in train_roma:
            preds.append(train_roma[img_id])
        train_combined[img_id] = preds

    # Create ensemble
    train_ensemble = create_ensemble(train_combined)

    # Count how many sources per prediction
    sources_count = {}
    for img_id, preds in train_combined.items():
        n = len(preds)
        sources_count[n] = sources_count.get(n, 0) + 1

    print(f"\nEnsemble statistics:")
    for n_sources in sorted(sources_count.keys(), reverse=True):
        print(f"  {sources_count[n_sources]} images with {n_sources} sources")

    # Save train ensemble
    train_output = root / "train_predictions_ensemble_rough_se2_roma.csv"
    save_csv(train_output, train_ensemble)
    print(f"\n✓ Saved: {train_output.name}")

    # Evaluate train ensemble
    result = evaluate_predictions(train_ensemble, gt)
    if result:
        score, accuracies, stats = result
        print(f"\nTrain Evaluation:")
        print(f"  Score: {score:.2f}")
        print(f"  @5m:   {accuracies['5m']:.2f}%")
        print(f"  @25m:  {accuracies['25m']:.2f}%")
        print(f"  @100m: {accuracies['100m']:.2f}%")
        print(f"  Median error: {stats['median']:.1f}px")

    # ===== TEST PREDICTIONS =====
    print("\n" + "="*80)
    print("TEST PREDICTIONS")
    print("-" * 80)

    # Load rough predictions
    test_rough_path = repo_root / "rough_matching" / "test_predicted.csv"
    test_rough = load_csv(test_rough_path)
    print(f"✓ Loaded rough: {len(test_rough)} predictions")

    # Load se2loftr
    test_se2_path = root / "test_predicted_se2loftr.csv"
    test_se2 = load_csv(test_se2_path)
    print(f"✓ Loaded se2loftr: {len(test_se2)} predictions")

    # Load minima-roma and apply -3px offset
    test_roma_path = root / "test_predicted_minima-roma.csv"
    test_roma_orig = load_csv(test_roma_path)
    test_roma = apply_y_offset(test_roma_orig, -3)
    print(f"✓ Loaded minima-roma: {len(test_roma)} predictions (applied -3px y-offset)")

    # Combine all predictions per image
    all_test_ids = set(test_rough.keys()) | set(test_se2.keys()) | set(test_roma.keys())
    test_combined = {}

    for img_id in all_test_ids:
        preds = []
        if img_id in test_rough:
            preds.append(test_rough[img_id])
        if img_id in test_se2:
            preds.append(test_se2[img_id])
        if img_id in test_roma:
            preds.append(test_roma[img_id])
        test_combined[img_id] = preds

    # Create ensemble
    test_ensemble = create_ensemble(test_combined)

    # Count sources
    test_sources_count = {}
    for img_id, preds in test_combined.items():
        n = len(preds)
        test_sources_count[n] = test_sources_count.get(n, 0) + 1

    print(f"\nEnsemble statistics:")
    for n_sources in sorted(test_sources_count.keys(), reverse=True):
        print(f"  {test_sources_count[n_sources]} images with {n_sources} sources")

    # Save test ensemble
    test_output = root / "test_predicted_ensemble_rough_se2_roma.csv"
    save_csv(test_output, test_ensemble)
    print(f"\n✓ Saved: {test_output.name}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train ensemble: {len(train_ensemble)} predictions")
    print(f"Test ensemble:  {len(test_ensemble)} predictions")
    print(f"Method: Median of rough GPS + se2loftr + minima-roma(-3px)")
    print("="*80)


if __name__ == "__main__":
    main()
