"""
Refine rough position predictions using MASt3R learned feature matching.

For each image:
  1. Crop the map around the rough predicted position
  2. Match features using MASt3R (asymmetric dense matcher, handles cross-domain gap)
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
from tqdm import tqdm

try:
    from romav2 import RoMaV2
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    RoMaV2 = None

# MASt3R environment setup
MAST3R_DIR = Path.home() / "mast3r"
if MAST3R_DIR.exists():
    import sys
    sys.path.insert(0, str(MAST3R_DIR))
    _original_cwd = os.getcwd()
    os.chdir(str(MAST3R_DIR))

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    import torch
    MAST3R_AVAILABLE = True

    # Restore original working directory after imports
    if MAST3R_DIR.exists():
        os.chdir(_original_cwd)
except ImportError:
    MAST3R_AVAILABLE = False
    torch = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CROP_SIZE = 500          # px — map crop around rough prediction
MATCH_SIZE = 512         # px — resize both images to this before MASt3R (512=training res)
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
def refine_one_mast3r(img_id, rough_x, rough_y, full_map, image_dir, model, device):
    """Try to refine a single image using MASt3R. Returns (x, y, n_inliers) or None."""
    from PIL import Image
    import tempfile

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

    # 3. Pre-resize drone to 512px width (like test_one.py)
    h_orig, w_orig = drone.shape[:2]
    mast3r_w = MATCH_SIZE
    mast3r_scale = mast3r_w / w_orig
    drone_resized = cv2.resize(drone, (mast3r_w, int(h_orig * mast3r_scale)))

    # 4. BGR -> RGB for MASt3R
    drone_rgb = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(map_crop, cv2.COLOR_BGR2RGB)

    # Save pre-resized dimensions (test_one.py uses these for center calculation)
    h_pre, w_pre = drone_resized.shape[:2]

    # 5. Convert to PIL and save to temp files (load_images expects paths)
    drone_pil = Image.fromarray(drone_rgb)
    crop_pil = Image.fromarray(crop_rgb)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_drone:
        drone_pil.save(tmp_drone.name, 'JPEG')
        drone_path = tmp_drone.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_crop:
        crop_pil.save(tmp_crop.name, 'JPEG')
        crop_path = tmp_crop.name

    try:
        # 6. Load images and run MASt3R inference
        images = load_images([drone_path, crop_path], size=MATCH_SIZE)

        with torch.inference_mode():
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()

        # 7. Find 2D-2D matches using fast reciprocal nearest neighbors
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8,
            device=device, dist='dot', block_size=2**13
        )

        # 8. Get actual image shapes from MASt3R output
        H0, W0 = view1['true_shape'][0]
        H1, W1 = view2['true_shape'][0]

        # DEBUG
        if img_id in [13, 14, 15]:
            print(f"\n[DEBUG img {img_id}] Drone true_shape: {W0}x{H0}, Crop true_shape: {W1}x{H1}")

        # Filter by border (3px)
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]

        # DEBUG
        if img_id in [13, 14, 15]:
            print(f"[DEBUG img {img_id}] Matches after border filter: {len(matches_im0)}")

        if len(matches_im0) < 4:
            return None

        # 9. Compute homography directly in true_shape coordinates (no scaling!)
        # This preserves aspect ratios and geometric relationships
        src_pts = matches_im0.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = matches_im1.reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        n_inliers = int(mask.sum())

        # DEBUG
        if img_id in [13, 14, 15]:
            print(f"[DEBUG img {img_id}] RANSAC inliers: {n_inliers}")

        if n_inliers < MIN_INLIERS:
            return None

        # 10. Project drone center through homography
        # Use pre-resized dimensions (matching test_one.py approach)
        center_x = float(w_pre) / 2
        center_y = float(h_pre) / 2
        center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(center, H)
        crop_x, crop_y = projected[0, 0]

        # 11. Convert to world coordinates (no scaling, matching test_one.py)
        world_x = crop_x + x0
        world_y = crop_y + y0

        # DEBUG
        if img_id in [13, 14, 15]:
            print(f"[DEBUG img {img_id}] Pre-resized drone: {w_pre}x{h_pre}")
            print(f"[DEBUG img {img_id}] Center in pre-resized: ({center_x:.1f}, {center_y:.1f})")
            print(f"[DEBUG img {img_id}] Projected: ({crop_x:.1f}, {crop_y:.1f})")
            print(f"[DEBUG img {img_id}] Offset: x0={x0}, y0={y0}")
            print(f"[DEBUG img {img_id}] World coords: ({world_x:.1f}, {world_y:.1f})\n")

        # 12. Sanity check
        drift = math.sqrt((world_x - rough_x) ** 2 + (world_y - rough_y) ** 2)
        if drift > MAX_DRIFT:
            return None

        return (world_x, world_y, n_inliers)
    finally:
        # Clean up temporary files
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)


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

    # Set up MASt3R
    if not MAST3R_AVAILABLE:
        raise RuntimeError("MASt3R is not available. Please ensure it's installed in ~/mast3r")

    print("Loading MASt3R model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    print(f"  MASt3R ready (device={device})")

    refined = {}
    n_success = 0
    n_fallback = 0
    ids = sorted(rough.keys())

    print(f"\nProcessing {len(ids)} images...\n")
    for img_id in tqdm(ids, desc="Refining"):
        rough_x, rough_y = rough[img_id]
        result = refine_one_mast3r(img_id, rough_x, rough_y, full_map, image_dir, model, device)

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
