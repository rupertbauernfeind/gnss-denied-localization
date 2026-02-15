#!/usr/bin/env python3
"""
Plot prediction errors as arrows from predictions to ground truth.
Helps visualize where and how predictions deviate from ground truth.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    """Load CSV file into dict mapping image_id -> (x, y)."""
    result = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['id'])
            x = float(row['x_pixel'])
            y = float(row['y_pixel'])
            result[img_id] = (x, y)
    return result


def main():
    """Plot prediction errors for train set."""
    root = Path(__file__).parent.parent

    # Load data
    print("Loading data...")
    gt_dict = load_csv(root / "data" / "train_data" / "train_pos.csv")
    pred_dict = load_csv(Path(__file__).parent / "train_predictions_ensemble_rough_se2_roma.csv")

    print(f"Ground truth: {len(gt_dict)} images")
    print(f"Predictions:  {len(pred_dict)} images")

    # Collect errors for images with both GT and predictions
    arrows = []
    errors = []

    for img_id in sorted(gt_dict.keys()):
        if img_id not in pred_dict:
            continue

        gt_x, gt_y = gt_dict[img_id]
        pred_x, pred_y = pred_dict[img_id]

        # Arrow from prediction to ground truth
        dx = gt_x - pred_x
        dy = gt_y - pred_y
        error = np.sqrt(dx**2 + dy**2)

        arrows.append((pred_x, pred_y, dx, dy))
        errors.append(error)

    print(f"\nAnalyzing {len(arrows)} images with both GT and predictions")
    print(f"Mean error: {np.mean(errors):.1f}px")
    print(f"Median error: {np.median(errors):.1f}px")
    print(f"Max error: {np.max(errors):.1f}px")

    # Filter arrows to only show errors ≤ 150px
    MAX_ERROR_TO_PLOT = 150.0
    arrows_filtered = [(a, e) for a, e in zip(arrows, errors) if e <= MAX_ERROR_TO_PLOT]
    n_filtered = len(arrows) - len(arrows_filtered)

    print(f"\nFiltering for visualization:")
    print(f"  Showing: {len(arrows_filtered)} arrows (error ≤ {MAX_ERROR_TO_PLOT}px)")
    print(f"  Hidden:  {n_filtered} arrows (error > {MAX_ERROR_TO_PLOT}px)")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot filtered arrows
    for (x, y, dx, dy), err in arrows_filtered:
        # Color based on error magnitude (green=good, yellow=medium, red=bad)
        if err <= 25:  # Within 5m
            color = 'green'
            alpha = 0.6
        elif err <= 125:  # Within 25m
            color = 'orange'
            alpha = 0.5
        else:  # > 25m
            color = 'red'
            alpha = 0.5

        ax.arrow(x, y, dx, dy,
                head_width=20, head_length=30,
                fc=color, ec=color, alpha=alpha, length_includes_head=True)

    # Plot predictions as small dots (only for filtered arrows)
    pred_x = [a[0] for a, e in arrows_filtered]
    pred_y = [a[1] for a, e in arrows_filtered]
    ax.scatter(pred_x, pred_y, c='blue', s=5, alpha=0.3, label='Predictions')

    # Plot ground truth as small dots (only for filtered arrows)
    gt_x = [a[0] + a[2] for a, e in arrows_filtered]  # pred_x + dx = gt_x
    gt_y = [a[1] + a[3] for a, e in arrows_filtered]  # pred_y + dy = gt_y
    ax.scatter(gt_x, gt_y, c='black', s=5, alpha=0.3, label='Ground Truth')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Prediction Errors (arrows point from prediction to GT)\n'
                 f'Showing {len(arrows_filtered)}/{len(arrows)} arrows (errors ≤{MAX_ERROR_TO_PLOT}px only)\n'
                 f'Mean: {np.mean(errors):.1f}px, Median: {np.median(errors):.1f}px | '
                 f'Green: ≤25px, Orange: ≤125px, Red: >125px')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "error_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
