#!/usr/bin/env python3
"""
Plot test flight trajectory from rough GPS predictions.
Shows the flight path by connecting consecutive image IDs.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


# Map dimensions
MAP_W, MAP_H = 5000, 2500


def load_csv_ordered(path):
    """Load CSV file as list of (img_id, x, y) tuples, sorted by img_id."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['id'])
            x = float(row['x_pixel'])
            y = float(row['y_pixel'])
            data.append((img_id, x, y))

    # Sort by img_id
    data.sort(key=lambda item: item[0])
    return data


def main():
    """Plot test trajectory with connected flight path."""
    root = Path(__file__).parent.parent

    # Load test rough GPS predictions
    print("Loading test rough GPS predictions...")
    test_path = root / "rough_matching" / "test_predicted.csv"

    if not test_path.exists():
        print(f"✗ Test predictions not found at {test_path}")
        return

    # Load data sorted by image ID
    test_data = load_csv_ordered(test_path)

    print(f"Loaded {len(test_data)} test predictions")
    print(f"ID range: {test_data[0][0]} to {test_data[-1][0]}")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Try to load and display map as background
    map_path = root / "data" / "map.png"
    if map_path.exists():
        print("Loading map image...")
        map_img = cv2.imread(str(map_path))
        if map_img is not None:
            # Convert BGR to RGB
            map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
            # Display map as background
            ax.imshow(map_img_rgb, extent=[0, MAP_W, MAP_H, 0], aspect='auto', alpha=0.5)
            print(f"Map loaded: {map_img.shape[1]}x{map_img.shape[0]}")

    # Draw lines connecting consecutive IDs
    n_segments = 0
    for i in range(len(test_data) - 1):
        img_id1, x1, y1 = test_data[i]
        img_id2, x2, y2 = test_data[i + 1]

        # Connect if IDs are consecutive
        if img_id2 == img_id1 + 1:
            ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=1)
            n_segments += 1

    # Plot all points
    x_coords = [item[1] for item in test_data]
    y_coords = [item[2] for item in test_data]
    ax.scatter(x_coords, y_coords, c='red', s=15, alpha=0.7,
               label=f'Test predictions ({len(test_data)})', zorder=5)

    # Mark start and end points
    if test_data:
        start_id, start_x, start_y = test_data[0]
        end_id, end_x, end_y = test_data[-1]
        ax.scatter([start_x], [start_y], c='green', s=100, marker='o',
                   label=f'Start (ID {start_id})', zorder=10, edgecolors='black', linewidths=2)
        ax.scatter([end_x], [end_y], c='orange', s=100, marker='s',
                   label=f'End (ID {end_id})', zorder=10, edgecolors='black', linewidths=2)

    # Set map bounds
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(MAP_H, 0)  # Invert y-axis to match image coordinates

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Test Flight Trajectory (Rough GPS)\n'
                 f'{len(test_data)} predictions | {n_segments} connected segments | '
                 f'IDs: {test_data[0][0]}-{test_data[-1][0]}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "test_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    print(f"  Connected {n_segments}/{len(test_data)-1} consecutive image pairs")

    plt.show()


if __name__ == "__main__":
    main()
