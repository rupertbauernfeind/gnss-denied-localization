#!/usr/bin/env python3
"""
Generate train and test predictions for minima-roma with -5px y-offset.

Usage:
    python generate_roma_y5_test.py
"""

import csv
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


def main():
    root = Path(__file__).parent

    print("="*80)
    print("GENERATE MINIMA-ROMA WITH -5PX Y-OFFSET (BEST METHOD)")
    print("="*80)

    # TRAIN predictions
    print("\nTRAIN PREDICTIONS")
    print("-" * 80)

    train_path = root / "train_predictions_minima-roma.csv"
    if not train_path.exists():
        print(f"✗ Missing: {train_path}")
        return

    train_pred = load_csv(train_path)
    print(f"✓ Loaded minima-roma train: {len(train_pred)} predictions")

    # Apply -5px offset
    train_pred_offset = apply_y_offset(train_pred, -5)

    # Save
    train_output_path = root / "train_predictions_minima-roma_y-5.csv"
    save_csv(train_output_path, train_pred_offset)
    print(f"✓ Saved: {train_output_path.name}")

    # TEST predictions
    print("\nTEST PREDICTIONS")
    print("-" * 80)

    test_path = root / "test_predicted_minima-roma.csv"
    if not test_path.exists():
        print(f"✗ Missing: {test_path}")
        return

    test_pred = load_csv(test_path)
    print(f"✓ Loaded minima-roma test: {len(test_pred)} predictions")

    # Apply -5px offset
    test_pred_offset = apply_y_offset(test_pred, -5)

    # Save
    test_output_path = root / "test_predicted_minima-roma_y-5.csv"
    save_csv(test_output_path, test_pred_offset)
    print(f"✓ Saved: {test_output_path.name}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Train: {len(train_pred_offset)} predictions")
    print(f"Test:  {len(test_pred_offset)} predictions")
    print("Applied: -5px y-offset to all predictions")
    print("="*80)


if __name__ == "__main__":
    main()
