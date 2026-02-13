"""
Evaluate predictions against ground truth using the competition metric.

Metric: mean of accuracies at three thresholds:
  - 5m  (25 px)
  - 25m (125 px)
  - 100m (500 px)

Usage:
    python evaluate.py
"""

import csv
import math
from pathlib import Path


def load_csv(path):
    """Load CSV into dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def main():
    root = Path(__file__).parent
    repo_root = root.parent
    gt = load_csv(repo_root / "data" / "train_data" / "train_pos.csv")
    pred = load_csv(root / "train_predictions.csv")

    common_ids = sorted(set(gt) & set(pred))
    print(f"Evaluating {len(common_ids)} samples\n")

    distances = []
    for i in common_ids:
        dx = pred[i][0] - gt[i][0]
        dy = pred[i][1] - gt[i][1]
        distances.append(math.sqrt(dx * dx + dy * dy))

    thresholds = [(25, "5m"), (125, "25m"), (500, "100m")]
    accuracies = []
    for px_thresh, label in thresholds:
        acc = sum(1 for d in distances if d <= px_thresh) / len(distances) * 100
        accuracies.append(acc)
        print(f"  Accuracy @ {label:>4} ({px_thresh:>3} px): {acc:6.2f}%")

    score = sum(accuracies) / len(accuracies)
    print(f"\n  Final score: {score:.2f}")

    # Extra stats
    distances.sort()
    print(f"\n  Distance stats (px):")
    print(f"    median: {distances[len(distances)//2]:.1f}")
    print(f"    mean:   {sum(distances)/len(distances):.1f}")
    print(f"    max:    {distances[-1]:.1f}")


if __name__ == "__main__":
    main()
