#!/usr/bin/env python3
"""
Generate train and test predictions for NewTop3 Median f150 ensemble.

Usage:
    python generate_newtop3_test.py
"""

import csv
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


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


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


def create_ensemble(predictions, rough_pred, max_deviation=150.0):
    """Create ensemble predictions using median aggregation."""
    ensemble_pred = {}

    all_ids = set()
    for model_preds in predictions.values():
        all_ids.update(model_preds.keys())

    stats = {'used': 0, 'fallback': 0, 'total': 0}

    for img_id in sorted(all_ids):
        stats['total'] += 1

        valid_preds = filter_valid_predictions(predictions, rough_pred, img_id,
                                               max_deviation)
        result = ensemble_median(valid_preds)

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
    models = ['minima-roma', 'loftr', 'xoftr']

    print("="*80)
    print("GENERATE NEWTOP3 MEDIAN F150 TRAIN & TEST PREDICTIONS")
    print("="*80)
    print(f"Models: {', '.join(models)}\n")

    # Process TRAIN predictions
    print("TRAIN PREDICTIONS")
    print("-" * 80)

    # Load train rough predictions
    train_rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    train_rough_pred = load_csv(train_rough_path)
    print(f"✓ Loaded train rough predictions: {len(train_rough_pred)}")

    # Load train predictions from models
    train_predictions = {}
    for model in models:
        pred_path = root / f"train_predictions_{model}.csv"
        if not pred_path.exists():
            print(f"✗ Missing: {pred_path}")
            return
        train_predictions[model] = load_csv(pred_path)
        print(f"✓ Loaded {model}: {len(train_predictions[model])} predictions")

    # Create train ensemble
    train_ensemble, train_stats = create_ensemble(
        train_predictions, train_rough_pred, max_deviation=150.0
    )

    # Save train ensemble
    train_output_path = root / "train_predictions_ensemble_newtop3_median_150.csv"
    save_csv(train_output_path, train_ensemble)
    print(f"\n✓ Saved train ensemble: {train_output_path.name}")
    print(f"  Ensembled: {train_stats['used']}/{train_stats['total']}, "
          f"Fallbacks: {train_stats['fallback']}")

    # Process TEST predictions
    print("\n" + "="*80)
    print("TEST PREDICTIONS")
    print("-" * 80)

    # Load test rough predictions
    test_rough_path = repo_root / "rough_matching" / "test_predicted.csv"
    test_rough_pred = load_csv(test_rough_path)
    print(f"✓ Loaded test rough predictions: {len(test_rough_pred)}")

    # Load test predictions from models
    test_predictions = {}
    for model in models:
        pred_path = root / f"test_predicted_{model}.csv"
        if not pred_path.exists():
            print(f"✗ Missing: {pred_path}")
            print(f"  Looking for: {pred_path}")
            return
        test_predictions[model] = load_csv(pred_path)
        print(f"✓ Loaded {model}: {len(test_predictions[model])} predictions")

    # Create test ensemble
    test_ensemble, test_stats = create_ensemble(
        test_predictions, test_rough_pred, max_deviation=150.0
    )

    # Save test ensemble
    test_output_path = root / "test_predicted_ensemble_newtop3_median_150.csv"
    save_csv(test_output_path, test_ensemble)
    print(f"\n✓ Saved test ensemble: {test_output_path.name}")
    print(f"  Ensembled: {test_stats['used']}/{test_stats['total']}, "
          f"Fallbacks: {test_stats['fallback']}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train: {len(train_ensemble)} predictions")
    print(f"Test:  {len(test_ensemble)} predictions")
    print("="*80)


if __name__ == "__main__":
    main()
