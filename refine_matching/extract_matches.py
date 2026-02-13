#!/usr/bin/env python3
"""
Extract and visualize MASt3R matches for detailed analysis.

Usage:
    python extract_matches.py
"""

import sys
import os
from pathlib import Path
import csv
import cv2
import numpy as np

# MASt3R environment setup (from refine.py lines 34-56)
MAST3R_DIR = Path.home() / "mast3r"
if MAST3R_DIR.exists():
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

    if MAST3R_DIR.exists():
        os.chdir(_original_cwd)
except ImportError:
    raise RuntimeError("MASt3R not available")

# Constants (from refine.py)
CROP_SIZE = 500
MATCH_SIZE = 512
MAP_W, MAP_H = 5000, 2500

def extract_and_visualize(img_id, rough_x, rough_y, full_map, image_dir, model, device, output_dir):
    """Extract matches and create visualization for one image."""

    # 1. Create output subdirectory
    img_dir = output_dir / f"img_{img_id:04d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract map crop (from refine.py lines 104-108)
    cx, cy = int(round(rough_x)), int(round(rough_y))
    x0 = max(0, min(MAP_W - CROP_SIZE, cx - CROP_SIZE // 2))
    y0 = max(0, min(MAP_H - CROP_SIZE, cy - CROP_SIZE // 2))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

    # 3. Load and pre-resize drone image (from refine.py lines 111-127)
    img_path = image_dir / f"{img_id:04d}.JPG"
    drone = cv2.imread(str(img_path))
    if drone is None:
        return False

    h_orig, w_orig = drone.shape[:2]
    mast3r_scale = MATCH_SIZE / w_orig
    drone_resized = cv2.resize(drone, (MATCH_SIZE, int(h_orig * mast3r_scale)))

    drone_rgb = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(map_crop, cv2.COLOR_BGR2RGB)

    h_pre, w_pre = drone_resized.shape[:2]

    # 4. Save input images
    cv2.imwrite(str(img_dir / "drone.jpg"), drone_resized)
    cv2.imwrite(str(img_dir / "map_crop.jpg"), map_crop)

    # 5. Run MASt3R inference (from refine.py lines 130-162)
    from PIL import Image
    import tempfile

    drone_pil = Image.fromarray(drone_rgb)
    crop_pil = Image.fromarray(crop_rgb)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_drone:
        drone_pil.save(tmp_drone.name, 'JPEG')
        drone_path = tmp_drone.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_crop:
        crop_pil.save(tmp_crop.name, 'JPEG')
        crop_path = tmp_crop.name

    try:
        images = load_images([drone_path, crop_path], size=MATCH_SIZE, square_ok=True)

        with torch.inference_mode():
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8,
            device=device, dist='dot', block_size=2**13
        )

        # 6. Border filtering (from refine.py lines 160-181)
        H0, W0 = view1['true_shape'][0]
        H1, W1 = view2['true_shape'][0]

        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]

        if len(matches_im0) < 4:
            return False

        # Convert to numpy arrays once (reuse everywhere)
        # Check if already numpy array or torch tensor
        if isinstance(matches_im0, np.ndarray):
            matches_im0_np = matches_im0
            matches_im1_np = matches_im1
        else:
            matches_im0_np = matches_im0.cpu().numpy()
            matches_im1_np = matches_im1.cpu().numpy()

        # 7. RANSAC homography (from refine.py lines 186-202)
        src_pts = matches_im0_np.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = matches_im1_np.reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return False

        inlier_mask = mask.ravel().astype(bool)
        inlier_indices = np.where(inlier_mask)[0]
        n_inliers = len(inlier_indices)

        print(f"Image {img_id}: {len(matches_im0_np)} matches, {n_inliers} inliers")

        # 8. Save match data
        np.savez(
            str(img_dir / "matches.npz"),
            matches_im0=matches_im0_np,
            matches_im1=matches_im1_np,
            H=H,
            mask=mask,
            inlier_indices=inlier_indices,
            true_shape_drone=(H0, W0),
            true_shape_crop=(H1, W1),
            pre_resized_shape=(h_pre, w_pre),
            crop_offset=(x0, y0)
        )

        # 9. Create visualization with 20 random inlier matches
        n_sample = min(20, n_inliers)
        sample_indices = np.random.choice(inlier_indices, n_sample, replace=False)

        # Create side-by-side visualization using true_shape dimensions (no scaling needed!)
        # Resize display images to match true_shape so coordinates line up perfectly
        drone_display = cv2.resize(drone_rgb, (int(W0), int(H0)))
        crop_display = cv2.resize(crop_rgb, (int(W1), int(H1)))

        h_d, w_d = drone_display.shape[:2]
        h_c, w_c = crop_display.shape[:2]
        max_h = max(h_d, h_c)

        # DEBUG: Print coordinate ranges
        print(f"  Drone display dims: {w_d}x{h_d} (matches true_shape)")
        print(f"  Crop display dims: {w_c}x{h_c} (matches true_shape)")
        print(f"  Drone matches - X: [{matches_im0_np[:, 0].min():.1f}, {matches_im0_np[:, 0].max():.1f}], Y: [{matches_im0_np[:, 1].min():.1f}, {matches_im0_np[:, 1].max():.1f}]")
        print(f"  Crop matches - X: [{matches_im1_np[:, 0].min():.1f}, {matches_im1_np[:, 0].max():.1f}], Y: [{matches_im1_np[:, 1].min():.1f}, {matches_im1_np[:, 1].max():.1f}]")

        # Convert to BGR for OpenCV
        img_left = cv2.cvtColor(drone_display, cv2.COLOR_RGB2BGR)
        img_right = cv2.cvtColor(crop_display, cv2.COLOR_RGB2BGR)

        # Pad to match heights
        if h_d < max_h:
            img_left = cv2.copyMakeBorder(img_left, 0, max_h - h_d, 0, 0, cv2.BORDER_CONSTANT)
        if h_c < max_h:
            img_right = cv2.copyMakeBorder(img_right, 0, max_h - h_c, 0, 0, cv2.BORDER_CONSTANT)

        vis = np.hstack([img_left, img_right])

        # Draw sampled matches with random colors - no scaling needed since images match true_shape!
        for i in sample_indices:
            x1 = int(matches_im0_np[i, 0])
            y1 = int(matches_im0_np[i, 1])
            x2 = int(matches_im1_np[i, 0])
            y2 = int(matches_im1_np[i, 1])

            pt1 = (x1, y1)
            pt2 = (x2 + w_d, y2)  # Offset by drone width for side-by-side layout

            # Random color for each match (BGR format)
            color = (
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))
            )

            # Draw line and circles with the same color
            cv2.line(vis, pt1, pt2, color, 2)
            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)

        # Add text overlay
        text = f"Image {img_id}: {n_sample}/{n_inliers} inliers shown"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(str(img_dir / "visualization.png"), vis)

        return True

    finally:
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)

def main():
    # Paths
    root = Path(__file__).parent
    repo_root = root.parent

    rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    image_dir = repo_root / "data" / "train_data" / "train_images"
    map_path = repo_root / "data" / "map.png"
    output_dir = root / "match_data"

    # Load data
    print("Loading data...")
    full_map = cv2.imread(str(map_path))
    assert full_map is not None

    rough = {}
    with open(rough_path) as f:
        for row in csv.DictReader(f):
            rough[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))

    # Load MASt3R model
    print("Loading MASt3R model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    model.eval()
    print(f"  MASt3R ready (device={device})")

    # Process images 13-22
    test_ids = list(range(13, 23))
    print(f"\nExtracting matches for images {test_ids[0]}-{test_ids[-1]}...\n")

    for img_id in test_ids:
        rough_x, rough_y = rough[img_id]
        success = extract_and_visualize(
            img_id, rough_x, rough_y, full_map, image_dir, model, device, output_dir
        )

        if success:
            print(f"  ✓ Saved data to {output_dir / f'img_{img_id:04d}'}")
        else:
            print(f"  ✗ Failed to process image {img_id}")

    print(f"\nDone! Data saved to {output_dir}")

if __name__ == "__main__":
    main()
