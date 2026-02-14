#!/usr/bin/env python3
"""
Plot train and test predictions on the map.
Shows spatial distribution of predictions bounded to map dimensions.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


# Map dimensions
MAP_W, MAP_H = 5000, 2500


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
    """Plot train and test predictions on map."""
    root = Path(__file__).parent.parent

    # Load predictions
    print("Loading predictions...")
    train_pred_path = Path(__file__).parent / "train_predictions.csv"
    test_pred_path = Path(__file__).parent / "test_predicted.csv"

    if not train_pred_path.exists():
        print(f"✗ Train predictions not found at {train_pred_path}")
        return

    if not test_pred_path.exists():
        print(f"✗ Test predictions not found at {test_pred_path}")
        return

    train_dict = load_csv(train_pred_path)
    test_dict = load_csv(test_pred_path)

    print(f"Train predictions: {len(train_dict)}")
    print(f"Test predictions:  {len(test_dict)}")

    # Extract coordinates
    train_x = [pos[0] for pos in train_dict.values()]
    train_y = [pos[1] for pos in train_dict.values()]
    test_x = [pos[0] for pos in test_dict.values()]
    test_y = [pos[1] for pos in test_dict.values()]

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

    # Plot predictions
    ax.scatter(train_x, train_y, c='blue', s=10, alpha=0.6, label=f'Train ({len(train_dict)})')
    ax.scatter(test_x, test_y, c='red', s=10, alpha=0.6, label=f'Test ({len(test_dict)})')

    # Set map bounds
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(MAP_H, 0)  # Invert y-axis to match image coordinates

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Train and Test Predictions on Map\n'
                 f'Train (blue): {len(train_dict)} predictions | Test (red): {len(test_dict)} predictions')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "predictions_map.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
