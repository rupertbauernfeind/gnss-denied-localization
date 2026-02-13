"""
Predict 2D map pixel positions from 3D camera coordinates in XMP metadata.

Fits an affine transform on training data (discarding height/z), then applies
it to both train and test sets.

Usage:
    python xmp_position_predict.py
"""

import csv
import os
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression


def parse_xmp_position(filepath):
    """Extract xcr:Position (x, y, z) from an XMP file.

    Handles two formats from Capturing Reality:
    1. Child element:  <xcr:Position>x y z</xcr:Position>
    2. Attribute:      xcr:Position="x y z"
    """
    with open(filepath, "r") as f:
        content = f.read()

    match = re.search(r"<xcr:Position>(.*?)</xcr:Position>", content)
    if not match:
        match = re.search(r'xcr:Position="(.*?)"', content)

    if match:
        return [float(c) for c in match.group(1).strip().split()]
    return None


def load_xmp_positions(image_dir):
    """Load 3D positions from all XMP files in a directory.
    Returns dict mapping image ID (int) -> [x, y, z].
    """
    positions = {}
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(".xmp"):
            continue
        image_id = int(filename.split(".")[0])
        pos = parse_xmp_position(str(image_dir / filename))
        if pos is not None:
            positions[image_id] = pos
        else:
            print(f"  WARNING: No position found in {filename}")
    return positions


def load_ground_truth(csv_path):
    """Load ground truth 2D positions from train_pos.csv.
    Returns dict mapping image ID (int) -> (x_pixel, y_pixel).
    """
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return gt


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    train_xmp_dir = root / "realityscan_positions" / "train"
    test_xmp_dir = root / "realityscan_positions" / "test"
    train_pos_csv = repo_root / "data" / "train_data" / "train_pos.csv"

    # 1. Load data
    print("Loading XMP positions...")
    train_3d = load_xmp_positions(train_xmp_dir)
    test_3d = load_xmp_positions(test_xmp_dir)
    print(f"  Train: {len(train_3d)} XMP files")
    print(f"  Test:  {len(test_3d)} XMP files")

    print("\nLoading ground truth...")
    ground_truth = load_ground_truth(train_pos_csv)
    print(f"  Ground truth: {len(ground_truth)} entries")

    # 2. Match train IDs and build arrays (using only x, y — discarding z)
    #    Filter out high-altitude images whose ground-plane (x,y) are unreliable
    z_threshold = 10.0
    common_ids = sorted(set(train_3d) & set(ground_truth))
    filtered_ids = [i for i in common_ids if train_3d[i][2] < z_threshold]
    excluded = len(common_ids) - len(filtered_ids)
    print(f"\n  Matched: {len(common_ids)} train samples with both XMP and ground truth")
    if excluded > 0:
        print(f"  Excluded {excluded} high-altitude images (z >= {z_threshold}):")
        for i in common_ids:
            if train_3d[i][2] >= z_threshold:
                print(f"    ID {i}: z = {train_3d[i][2]:.1f}")
    print(f"  Using {len(filtered_ids)} samples for fitting")

    X_train = np.array([[train_3d[i][0], train_3d[i][1]] for i in filtered_ids])
    Y_train = np.array([[ground_truth[i][0], ground_truth[i][1]] for i in filtered_ids])

    # 3. Fit affine transform: (x, y) -> x_pixel and (x, y) -> y_pixel
    model_x = LinearRegression().fit(X_train, Y_train[:, 0])
    model_y = LinearRegression().fit(X_train, Y_train[:, 1])

    # 4. Diagnostics
    pred_x_train = model_x.predict(X_train)
    pred_y_train = model_y.predict(X_train)
    residuals = np.sqrt((pred_x_train - Y_train[:, 0]) ** 2 + (pred_y_train - Y_train[:, 1]) ** 2)

    print(f"\n{'=' * 60}")
    print("AFFINE TRANSFORM (z discarded)")
    print(f"{'=' * 60}")
    print(f"  x_pixel = {model_x.coef_[0]:.6f} * x + {model_x.coef_[1]:.6f} * y + {model_x.intercept_:.6f}")
    print(f"  y_pixel = {model_y.coef_[0]:.6f} * x + {model_y.coef_[1]:.6f} * y + {model_y.intercept_:.6f}")

    print(f"\n{'=' * 60}")
    print("TRAINING FIT DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"  R-squared:  x = {model_x.score(X_train, Y_train[:, 0]):.6f}  y = {model_y.score(X_train, Y_train[:, 1]):.6f}")
    print(f"  RMSE:       x = {np.sqrt(np.mean((pred_x_train - Y_train[:, 0]) ** 2)):.2f} px  y = {np.sqrt(np.mean((pred_y_train - Y_train[:, 1]) ** 2)):.2f} px")
    print(f"  Euclidean:  mean = {np.mean(residuals):.2f} px  max = {np.max(residuals):.2f} px  median = {np.median(residuals):.2f} px")

    # 5. Export predictions
    print(f"\n{'=' * 60}")
    print("EXPORTING PREDICTIONS")
    print(f"{'=' * 60}")

    for label, positions, output_name in [
        ("Train", train_3d, "train_predictions.csv"),
        ("Test", test_3d, "test_predicted.csv"),
    ]:
        ids = sorted(positions.keys())
        X = np.array([[positions[i][0], positions[i][1]] for i in ids])
        px = model_x.predict(X)
        py = model_y.predict(X)

        output_path = root / output_name
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x_pixel", "y_pixel"])
            for j, image_id in enumerate(ids):
                writer.writerow([image_id, px[j], py[j]])

        print(f"  {label}: {len(ids)} predictions -> {output_path.name}")
        print(f"    x_pixel range: [{px.min():.1f}, {px.max():.1f}]")
        print(f"    y_pixel range: [{py.min():.1f}, {py.max():.1f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
