"""
Refine rough position predictions using RoMa v2 learned feature matching.

For each image:
  1. Crop the map around the rough predicted position
  2. Match features using RoMa v2 (dense matcher, handles cross-domain gap)
  3. Compute a homography (RANSAC) and project the image center
     onto the map to get a refined position

Usage:
    python refine.py              # train (default)
    python refine.py --split test # test
"""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # skip torch.compile (no MSVC/GPU)

import cv2
import numpy as np
from romav2 import RoMaV2
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CROP_SIZE = 500          # px — map crop around rough prediction
MATCH_SIZE = 256         # px — resize both images to this before RoMa (CPU speed)
NUM_MATCHES = 5000       # number of correspondences to sample from dense match
MIN_INLIERS = 20         # minimum RANSAC inliers to accept refinement
MAX_DRIFT = 250          # px — discard refinement if it moves too far from rough
MAP_W, MAP_H = 5000, 2500


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load CSV -> dict: id -> (x_pixel, y_pixel)."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def save_csv(path, predictions):
    """Save dict {id: (x, y)} to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_pixel", "y_pixel"])
        for img_id in sorted(predictions):
            x, y = predictions[img_id]
            writer.writerow([img_id, x, y])


def id_to_filename(img_id):
    return f"{img_id:04d}.JPG"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def refine_one(img_id, rough_x, rough_y, full_map, image_dir, matcher):
    """Try to refine a single image. Returns (x, y, n_inliers) or None."""

    # 1. Crop map around rough prediction (full res for accuracy)
    cx = int(round(rough_x))
    cy = int(round(rough_y))
    x0 = max(0, min(MAP_W - CROP_SIZE, cx - CROP_SIZE // 2))
    y0 = max(0, min(MAP_H - CROP_SIZE, cy - CROP_SIZE // 2))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

    # 2. Load drone image
    img_path = str(image_dir / id_to_filename(img_id))
    drone = cv2.imread(img_path)
    if drone is None:
        return None

    # 3. Resize both to MATCH_SIZE for fast RoMa matching on CPU
    drone_match = cv2.resize(drone, (MATCH_SIZE, MATCH_SIZE))
    crop_match = cv2.resize(map_crop, (MATCH_SIZE, MATCH_SIZE))

    # 4. BGR -> RGB for RoMa
    drone_rgb = cv2.cvtColor(drone_match, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop_match, cv2.COLOR_BGR2RGB)

    # 5. RoMa v2 matching (at MATCH_SIZE resolution)
    preds = matcher.match(drone_rgb, crop_rgb)
    matches, confidence, _, _ = matcher.sample(preds, NUM_MATCHES)

    # 6. Convert to pixel coordinates (in MATCH_SIZE space)
    kp_drone, kp_crop = RoMaV2.to_pixel_coordinates(
        matches, MATCH_SIZE, MATCH_SIZE, MATCH_SIZE, MATCH_SIZE
    )
    kp0 = kp_drone.cpu().numpy()
    kp1 = kp_crop.cpu().numpy()

    if len(kp0) < 4:
        return None

    # 7. Scale keypoints back to original crop coords (CROP_SIZE x CROP_SIZE)
    scale_crop = CROP_SIZE / MATCH_SIZE
    kp1_full = kp1 * scale_crop

    # 8. Find homography: MATCH_SIZE drone -> full crop coords
    src_pts = kp0.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = kp1_full.reshape(-1, 1, 2).astype(np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None

    n_inliers = int(mask.sum())
    if n_inliers < MIN_INLIERS:
        return None

    # 9. Project drone image center through H -> map crop coords -> world coords
    center = np.float32([[MATCH_SIZE / 2, MATCH_SIZE / 2]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(center, H)
    crop_x, crop_y = projected[0, 0]

    world_x = crop_x + x0
    world_y = crop_y + y0

    # 10. Sanity check
    drift = math.sqrt((world_x - rough_x) ** 2 + (world_y - rough_y) ** 2)
    if drift > MAX_DRIFT:
        return None

    return (world_x, world_y, n_inliers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    root = Path(__file__).parent
    repo_root = root.parent

    if args.split == "train":
        rough_path = repo_root / "rough_matching" / "train_predictions.csv"
        image_dir = repo_root / "data" / "train_data" / "train_images"
        output_path = root / "train_predictions.csv"
    else:
        rough_path = repo_root / "rough_matching" / "test_predicted.csv"
        image_dir = repo_root / "data" / "test_data" / "test_images"
        output_path = root / "test_predictions.csv"

    map_path = repo_root / "data" / "map.png"

    print(f"Split: {args.split}")
    print("Loading data...")
    rough = load_csv(rough_path)
    full_map = cv2.imread(str(map_path))
    assert full_map is not None, f"Could not load map: {map_path}"
    print(f"  Map: {full_map.shape[1]}x{full_map.shape[0]}")
    print(f"  Rough predictions: {len(rough)}")

    # Set up RoMa v2
    print("Loading RoMa v2 model...")
    matcher = RoMaV2()
    print("  RoMa v2 ready")

    refined = {}
    n_success = 0
    n_fallback = 0
    ids = sorted(rough.keys())

    print(f"\nProcessing {len(ids)} images...\n")
    for img_id in tqdm(ids, desc="Refining"):
        rough_x, rough_y = rough[img_id]
        result = refine_one(img_id, rough_x, rough_y, full_map, image_dir, matcher)

        if result is not None:
            world_x, world_y, n_inliers = result
            refined[img_id] = (world_x, world_y)
            n_success += 1
            tqdm.write(f"  REFINED  id={img_id:<5} inliers={n_inliers:>3}")
        else:
            refined[img_id] = (rough_x, rough_y)
            n_fallback += 1

    print(f"\nDone: {n_success} refined, {n_fallback} fallback")

    save_csv(output_path, refined)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
