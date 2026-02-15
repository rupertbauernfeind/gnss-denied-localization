#!/usr/bin/env python3
"""
Apply offset to ensemble_rough_se2_roma test predictions.

Usage:
    python apply_ensemble_offset.py [x_offset] [y_offset]

Default: x=+2px, y=-3px (slightly overstated from optimal +1, -2)
"""

import csv
import sys
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


def apply_offset(predictions, x_offset=0, y_offset=0):
    """Apply constant offsets to coordinates."""
    result = {}
    for img_id, (x, y) in predictions.items():
        result[img_id] = (x + x_offset, y + y_offset)
    return result


def main():
    # Parse command line arguments
    x_offset = 2  # Default: slightly overstated
    y_offset = -3  # Default: slightly overstated

    if len(sys.argv) >= 3:
        x_offset = int(sys.argv[1])
        y_offset = int(sys.argv[2])

    root = Path(__file__).parent

    print("="*80)
    print("APPLYING OFFSET TO ENSEMBLE TEST PREDICTIONS")
    print("="*80)
    print(f"Offset: x={x_offset:+d}px, y={y_offset:+d}px")
    print()

    # Process train predictions (for verification)
    train_input = root / "train_predictions_ensemble_rough_se2_roma.csv"
    if train_input.exists():
        train_pred = load_csv(train_input)
        train_output = root / f"train_predictions_ensemble_rough_se2_roma_offset.csv"
        train_offset = apply_offset(train_pred, x_offset=x_offset, y_offset=y_offset)
        save_csv(train_output, train_offset)
        print(f"✓ Train: {train_input.name} → {train_output.name}")
        print(f"  {len(train_offset)} predictions with offset applied")
    else:
        print(f"✗ Train file not found: {train_input}")

    # Process test predictions
    test_input = root / "test_predicted_ensemble_rough_se2_roma.csv"
    if test_input.exists():
        test_pred = load_csv(test_input)
        test_output = root / f"test_predicted_ensemble_rough_se2_roma_offset.csv"
        test_offset = apply_offset(test_pred, x_offset=x_offset, y_offset=y_offset)
        save_csv(test_output, test_offset)
        print(f"✓ Test:  {test_input.name} → {test_output.name}")
        print(f"  {len(test_offset)} predictions with offset applied")
    else:
        print(f"✗ Test file not found: {test_input}")

    print()
    print("="*80)
    print("OFFSET APPLIED SUCCESSFULLY")
    print("="*80)
    print(f"Applied offset: x={x_offset:+d}px, y={y_offset:+d}px")
    print("Files saved:")
    print("  - train_predictions_ensemble_rough_se2_roma_offset.csv")
    print("  - test_predicted_ensemble_rough_se2_roma_offset.csv")
    print("="*80)


if __name__ == "__main__":
    main()
