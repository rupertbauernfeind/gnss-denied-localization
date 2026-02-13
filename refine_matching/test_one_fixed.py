"""
Quick test: try multiple matching strategies on ONE image.

Usage:
    python test_one.py [image_id]                  # Run all matchers including MASt3R
    python test_one.py --dry-run [image_id]        # Test MASt3R coordinate logic only (no model loading)
    python test_one.py --dry-run                   # Test all available pre-computed matches
"""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import kornia.feature as KF

# MASt3R environment setup
MAST3R_DIR = Path.home() / "mast3r"
if MAST3R_DIR.exists():
    import sys
    sys.path.insert(0, str(MAST3R_DIR))
    import os
    _original_cwd = os.getcwd()
    os.chdir(str(MAST3R_DIR))

try:
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    MAST3R_AVAILABLE = True

    # Restore original working directory after imports
    if MAST3R_DIR.exists():
        os.chdir(_original_cwd)
except ImportError:
    MAST3R_AVAILABLE = False

CROP_SIZE = 500
MAP_W, MAP_H = 5000, 2500


def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def try_sift(gray_drone, gray_crop, ratio=0.8):
    """SIFT + FLANN, return (src_pts, dst_pts, n_good, n_inliers, H, mask, kp1, kp2, good)."""
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)
    kp1, des1 = sift.detectAndCompute(gray_drone, None)
    kp2, des2 = sift.detectAndCompute(gray_crop, None)
    print(f"    SIFT keypoints: drone={len(kp1)}, crop={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])

    print(f"    Good matches (ratio={ratio}): {len(good)}")
    if len(good) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"    RANSAC inliers: {n_inliers}")
    return kp1, kp2, good, H, mask, n_inliers


def try_orb(gray_drone, gray_crop):
    """ORB + BFMatcher."""
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(gray_drone, None)
    kp2, des2 = orb.detectAndCompute(gray_crop, None)
    print(f"    ORB keypoints: drone={len(kp1)}, crop={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])

    print(f"    Good matches: {len(good)}")
    if len(good) < 4:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"    RANSAC inliers: {n_inliers}")
    return kp1, kp2, good, H, mask, n_inliers


def try_template_multiscale(gray_drone, gray_crop):
    """Resize drone image at multiple scales, template match against map crop."""
    best_val = -1
    best_loc = None
    best_scale = None

    # Try a range of scales — the drone image needs to be shrunk to roughly
    # match the map resolution. We don't know the exact scale so try many.
    for scale_pct in range(3, 20):  # 3% to 19% of original drone width
        scale = scale_pct / 100.0
        w = int(gray_drone.shape[1] * scale)
        h = int(gray_drone.shape[0] * scale)
        if w < 20 or h < 20 or w > gray_crop.shape[1] or h > gray_crop.shape[0]:
            continue

        resized = cv2.resize(gray_drone, (w, h))
        result = cv2.matchTemplate(gray_crop, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_w, best_h = w, h

    if best_loc is None:
        return None

    # Center of the best match in crop coords
    center_x = best_loc[0] + best_w / 2
    center_y = best_loc[1] + best_h / 2

    print(f"    Template match: scale={best_scale:.2f}, score={best_val:.3f}, "
          f"center=({center_x:.0f},{center_y:.0f})")
    return center_x, center_y, best_val, best_scale


def proc_to_orig_xy(xp, yp, orig_w, orig_h, size=512, patch_size=16, square_ok=True):
    """
    Convert coordinates from MASt3R's processed image space back to original image space.

    Mirrors the preprocessing done by dust3r.utils.image.load_images():
    1. Resize to target size
    2. Center-crop to make dimensions divisible by patch_size
    3. If square_ok=False and image is square, crop to 4:3 aspect ratio

    Args:
        xp, yp: Coordinates in processed (cropped) image space
        orig_w, orig_h: Original image dimensions
        size: Target size for resizing (default 512)
        patch_size: Patch size for dimension alignment (default 16)
        square_ok: Whether square images are kept square (default True)

    Returns:
        x_orig, y_orig: Coordinates in original image space
    """
    # Step 1: Compute resize scale based on max dimension
    S = max(orig_w, orig_h)
    scale = size / S
    w_res = int(round(orig_w * scale))
    h_res = int(round(orig_h * scale))

    # Step 2: Compute center-crop dimensions
    cx, cy = w_res // 2, h_res // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2

    # Step 3: Apply square→4:3 conversion if needed
    if (not square_ok) and (w_res == h_res):
        halfh = 3 * halfw / 4  # This is the key bug when square_ok=False

    # Compute crop offsets
    left = cx - halfw
    top = cy - halfh

    # Convert: processed → resized → original
    x_res = xp + left
    y_res = yp + top
    x_orig = x_res / scale
    y_orig = y_res / scale

    return x_orig, y_orig


def test_mast3r_dry_run(img_id, gt_x, gt_y, rough_err, match_data_dir):
    """
    Test MASt3R coordinate conversion using pre-computed matches from extract_matches.py.
    No need to import or run MASt3R - just tests the projection logic.

    Args:
        img_id: Image ID
        gt_x, gt_y: Ground truth coordinates
        rough_err: Rough matching baseline error
        match_data_dir: Path to match_data directory

    Returns:
        True if successful, False otherwise
    """
    match_file = match_data_dir / f"img_{img_id:04d}" / "matches.npz"

    if not match_file.exists():
        print(f"    No pre-computed matches found at {match_file}")
        return False

    print(f"\n  [DRY RUN] MASt3R (from pre-computed matches):")

    # Load pre-computed match data
    data = np.load(str(match_file))
    matches_im0 = data['matches_im0']  # drone matches
    matches_im1 = data['matches_im1']  # crop matches
    H = data['H']  # homography
    mask = data['mask']  # inlier mask
    H0, W0 = data['true_shape_drone']  # processed drone shape
    H1, W1 = data['true_shape_crop']   # processed crop shape
    x0, y0 = data['crop_offset']  # crop offset in world coords

    n_inliers = int(mask.sum())
    print(f"    Loaded: {len(matches_im0)} matches, {n_inliers} inliers")
    print(f"    Processed shapes - Drone: ({W0}, {H0}), Crop: ({W1}, {H1})")

    if n_inliers < 4:
        print(f"    Insufficient inliers ({n_inliers} < 4)")
        return False

    # Extract inlier matches
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp0 = matches_im0[inlier_mask]
    inlier_kp1 = matches_im1[inlier_mask]

    # METHOD 1: Center point projection
    center = np.float32([[W0 / 2, H0 / 2]]).reshape(-1, 1, 2)
    proj_center = cv2.perspectiveTransform(center, H)
    xp_center, yp_center = proj_center[0, 0]

    # METHOD 2: Median offset from all inliers (biased when matches cluster)
    offsets_x = inlier_kp1[:, 0] - inlier_kp0[:, 0]
    offsets_y = inlier_kp1[:, 1] - inlier_kp0[:, 1]
    median_offset_x = np.median(offsets_x)
    median_offset_y = np.median(offsets_y)

    center_x, center_y = W0 / 2, H0 / 2
    xp_offset = center_x + median_offset_x
    yp_offset = center_y + median_offset_y

    # METHOD 3: Median of transformed points (more robust)
    # Transform all inlier drone points through homography and take median
    inlier_kp0_reshaped = inlier_kp0.reshape(-1, 1, 2).astype(np.float32)
    transformed_pts = cv2.perspectiveTransform(inlier_kp0_reshaped, H).reshape(-1, 2)
    xp_median_transform = np.median(transformed_pts[:, 0])
    yp_median_transform = np.median(transformed_pts[:, 1])

    print(f"    Drone center (processed coords): ({center_x:.1f}, {center_y:.1f})")
    print(f"    Center method → ({xp_center:.1f}, {yp_center:.1f})")
    print(f"    Median offset method → ({xp_offset:.1f}, {yp_offset:.1f})")
    print(f"    Median transformed method → ({xp_median_transform:.1f}, {yp_median_transform:.1f})")

    # Test all three methods
    methods = [
        ("Center", (xp_center, yp_center)),
        ("MedOff", (xp_offset, yp_offset)),
        ("MedXfm", (xp_median_transform, yp_median_transform))
    ]

    for method_name, (xp, yp) in methods:
        # Convert from processed crop coords to original crop coords (500×500)
        x_in_crop, y_in_crop = proc_to_orig_xy(
            xp, yp,
            orig_w=500,  # map_crop.shape[1]
            orig_h=500,  # map_crop.shape[0]
            size=512,
            patch_size=16,
            square_ok=True
        )

        # Convert to world coordinates
        rx, ry = x0 + x_in_crop, y0 + y_in_crop
        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)

        print(f"    {method_name:6s} → crop: ({x_in_crop:.1f}, {y_in_crop:.1f}) → world: ({rx:.1f}, {ry:.1f})")
        print(f"             err={err:.1f}px ({err/5:.0f}m) {'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

    return True


def try_loftr(gray_drone, gray_crop, matcher, conf_thresh=0.5):
    """LoFTR learned matcher. Returns (src_pts, dst_pts, n_matches, H, mask, n_inliers)."""
    # Convert to float32 [0,1] tensors with shape (1, 1, H, W)
    t_drone = torch.from_numpy(gray_drone).float()[None, None] / 255.0
    t_crop = torch.from_numpy(gray_crop).float()[None, None] / 255.0

    with torch.inference_mode():
        result = matcher({"image0": t_drone, "image1": t_crop})

    kp0 = result["keypoints0"].cpu().numpy()
    kp1 = result["keypoints1"].cpu().numpy()
    conf = result["confidence"].cpu().numpy()

    print(f"    LoFTR raw matches: {len(kp0)}")

    # Filter by confidence
    mask_conf = conf >= conf_thresh
    kp0 = kp0[mask_conf]
    kp1 = kp1[mask_conf]
    conf = conf[mask_conf]
    print(f"    After confidence filter (>={conf_thresh}): {len(kp0)}")

    if len(kp0) < 4:
        return None

    src = kp0.reshape(-1, 1, 2).astype(np.float32)
    dst = kp1.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"    RANSAC inliers: {n_inliers}")

    if n_inliers < 4:
        return None

    return kp0, kp1, conf, H, mask, n_inliers


def try_mast3r(rgb_drone, rgb_crop, model, device):
    """MASt3R asymmetric learned matcher. Returns (kp0, kp1, None, H, mask, n_inliers)."""
    from PIL import Image
    import tempfile
    import os

    # Convert numpy RGB to PIL and save to temp files
    drone_pil = Image.fromarray(rgb_drone)
    crop_pil = Image.fromarray(rgb_crop)

    # Create temporary files for load_images (expects file paths)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_drone:
        drone_pil.save(tmp_drone.name, 'JPEG')
        drone_path = tmp_drone.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_crop:
        crop_pil.save(tmp_crop.name, 'JPEG')
        crop_path = tmp_crop.name

    try:
        # Load images (returns list with 'img' tensors)
        # IMPORTANT: square_ok=True prevents square→4:3 cropping
        images = load_images([drone_path, crop_path], size=512, square_ok=True)

        # Run inference to get dense descriptors
        with torch.inference_mode():
            output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1 = pred1['desc'].squeeze(0).detach()
        desc2 = pred2['desc'].squeeze(0).detach()

        # Find 2D-2D matches using fast reciprocal nearest neighbors
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8,
            device=device, dist='dot', block_size=2**13
        )

        print(f"    MASt3R raw matches: {len(matches_im0)}")

        # Filter by border (3px)
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]

        print(f"    After border filter: {len(matches_im0)}")

        # Debug: show match point ranges
        if len(matches_im0) > 0:
            print(f"    DEBUG: Drone matches - X: [{matches_im0[:, 0].min():.1f}, {matches_im0[:, 0].max():.1f}], "
                  f"Y: [{matches_im0[:, 1].min():.1f}, {matches_im0[:, 1].max():.1f}]")
            print(f"    DEBUG: Crop matches  - X: [{matches_im1[:, 0].min():.1f}, {matches_im1[:, 0].max():.1f}], "
                  f"Y: [{matches_im1[:, 1].min():.1f}, {matches_im1[:, 1].max():.1f}]")
            print(f"    DEBUG: Processed shapes - Drone: ({W0}, {H0}), Crop: ({W1}, {H1})")

        if len(matches_im0) < 4:
            return None

        # Compute homography (matches are already numpy arrays)
        src = matches_im0.reshape(-1, 1, 2).astype(np.float32)
        dst = matches_im1.reshape(-1, 1, 2).astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        n_inliers = int(mask.sum()) if mask is not None else 0
        print(f"    RANSAC inliers: {n_inliers}")

        if n_inliers < 4:
            return None

        # Return processed dimensions for coordinate conversion
        return matches_im0, matches_im1, None, H, mask, n_inliers, (W0, H0), (W1, H1)
    finally:
        # Clean up temporary files
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)


def main():
    root = Path(__file__).parent
    repo_root = root.parent
    match_data_dir = root / "match_data"

    rough = load_csv(repo_root / "rough_matching" / "train_predictions.csv")
    gt = load_csv(repo_root / "data" / "train_data" / "train_pos.csv")

    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    if dry_run:
        # Dry run mode: test coordinate conversion with pre-computed matches
        if args:
            # Single image specified
            img_ids = [int(args[0])]
        else:
            # Test all available pre-computed matches
            if not match_data_dir.exists():
                print(f"No match data directory found at {match_data_dir}")
                return
            img_dirs = sorted(match_data_dir.glob("img_*"))
            img_ids = [int(d.name.split("_")[1]) for d in img_dirs if (d / "matches.npz").exists()]
            if not img_ids:
                print(f"No pre-computed matches found in {match_data_dir}")
                return
            print(f"Found pre-computed matches for {len(img_ids)} images: {img_ids}")

        # Test each image in dry run mode
        for img_id in img_ids:
            if img_id not in gt:
                print(f"\nImage {img_id}: No ground truth available, skipping")
                continue

            rough_x, rough_y = rough[img_id]
            gt_x, gt_y = gt[img_id]
            rough_err = math.sqrt((rough_x - gt_x) ** 2 + (rough_y - gt_y) ** 2)

            print(f"\n{'='*60}")
            print(f"Image id: {img_id}")
            print(f"  GT:    ({gt_x:.1f}, {gt_y:.1f})")
            print(f"  Rough: ({rough_x:.1f}, {rough_y:.1f})  err={rough_err:.1f}px ({rough_err/5:.0f}m)")

            test_mast3r_dry_run(img_id, gt_x, gt_y, rough_err, match_data_dir)

        return

    # Normal mode: run full matching pipeline
    image_dir = repo_root / "data" / "train_data" / "train_images"
    full_map = cv2.imread(str(repo_root / "data" / "map.png"))
    assert full_map is not None

    img_id = int(args[0]) if args else sorted(set(rough) & set(gt))[0]

    rough_x, rough_y = rough[img_id]
    gt_x, gt_y = gt[img_id]
    rough_err = math.sqrt((rough_x - gt_x) ** 2 + (rough_y - gt_y) ** 2)

    print(f"Image id: {img_id}")
    print(f"  GT:    ({gt_x:.1f}, {gt_y:.1f})")
    print(f"  Rough: ({rough_x:.1f}, {rough_y:.1f})  err={rough_err:.1f}px ({rough_err/5:.0f}m)")

    # Crop map
    cx, cy = int(round(rough_x)), int(round(rough_y))
    x0 = max(0, min(MAP_W - CROP_SIZE, cx - CROP_SIZE // 2))
    y0 = max(0, min(MAP_H - CROP_SIZE, cy - CROP_SIZE // 2))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

    # Load drone image
    drone = cv2.imread(str(image_dir / f"{img_id:04d}.JPG"))
    assert drone is not None
    h_orig, w_orig = drone.shape[:2]
    print(f"  Drone image: {w_orig}x{h_orig}")

    # Resize drone for feature matching
    resize_w = 800
    scale_f = resize_w / w_orig
    drone_small = cv2.resize(drone, (resize_w, int(h_orig * scale_f)))

    gray_crop = cv2.cvtColor(map_crop, cv2.COLOR_BGR2GRAY)
    gray_drone = cv2.cvtColor(drone_small, cv2.COLOR_BGR2GRAY)
    gray_drone_full = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)

    # --- Strategy 1: SIFT (permissive ratio) ---
    print("\n  [1] SIFT (ratio=0.85, low contrast threshold):")
    sift_result = try_sift(gray_drone, gray_crop, ratio=0.85)

    if sift_result and sift_result[5] >= 6:
        H = sift_result[3]
        h_s, w_s = gray_drone.shape[:2]
        center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(center, H)
        rx, ry = proj[0, 0, 0] + x0, proj[0, 0, 1] + y0
        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

    # --- Strategy 2: ORB ---
    print("\n  [2] ORB:")
    orb_result = try_orb(gray_drone, gray_crop)

    if orb_result and orb_result[5] >= 6:
        H = orb_result[3]
        h_s, w_s = gray_drone.shape[:2]
        center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(center, H)
        rx, ry = proj[0, 0, 0] + x0, proj[0, 0, 1] + y0
        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

    # --- Strategy 3: Multi-scale template matching ---
    print("\n  [3] Template matching (multi-scale):")
    tmpl_result = try_template_multiscale(gray_drone_full, gray_crop)

    if tmpl_result:
        crop_cx, crop_cy, score, best_scale = tmpl_result
        rx, ry = crop_cx + x0, crop_cy + y0
        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

    # --- Strategy 4: LoFTR (learned matcher) ---
    print("\n  [4] LoFTR (outdoor):")
    matcher = KF.LoFTR(pretrained="outdoor")
    matcher.eval()
    # LoFTR works on resized drone (same as SIFT) — smaller = faster on CPU
    # Resize drone to ~500px wide (closer to crop size, faster on CPU)
    loftr_w = 500
    loftr_scale = loftr_w / w_orig
    drone_loftr = cv2.resize(drone, (loftr_w, int(h_orig * loftr_scale)))
    gray_drone_loftr = cv2.cvtColor(drone_loftr, cv2.COLOR_BGR2GRAY)
    loftr_result = try_loftr(gray_drone_loftr, gray_crop, matcher, conf_thresh=0.3)

    if loftr_result and loftr_result[5] >= 4:
        H = loftr_result[3]
        h_s, w_s = gray_drone_loftr.shape[:2]
        center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(center, H)
        rx, ry = proj[0, 0, 0] + x0, proj[0, 0, 1] + y0
        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

    # --- Strategy 5: MASt3R (learned asymmetric matcher) ---
    if MAST3R_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n  [5] MASt3R (asymmetric):")
        print(f"    Loading MASt3R model (device={device})...")
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        mast3r_model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        mast3r_model.eval()
        print("    MASt3R ready")

        # Resize to 512px for MASt3R
        mast3r_w = 512
        mast3r_scale = mast3r_w / w_orig
        drone_mast3r = cv2.resize(drone, (mast3r_w, int(h_orig * mast3r_scale)))

        # Convert to RGB
        rgb_drone = cv2.cvtColor(drone_mast3r, cv2.COLOR_BGR2RGB)
        rgb_crop = cv2.cvtColor(map_crop, cv2.COLOR_BGR2RGB)

        mast3r_result = try_mast3r(rgb_drone, rgb_crop, mast3r_model, device)

        if mast3r_result and mast3r_result[5] >= 4:
            H = mast3r_result[3]
            mask = mast3r_result[4]
            kp0 = mast3r_result[0]  # drone matches
            kp1 = mast3r_result[1]  # crop matches
            # Get processed dimensions from MASt3R
            drone_proc_shape = mast3r_result[6]  # (W0, H0)
            crop_proc_shape = mast3r_result[7]   # (W1, H1)

            # METHOD 1: Use center point
            W0, H0 = int(drone_proc_shape[0]), int(drone_proc_shape[1])
            center = np.float32([[W0 / 2, H0 / 2]]).reshape(-1, 1, 2)
            proj_center = cv2.perspectiveTransform(center, H)
            xp_center, yp_center = proj_center[0, 0]

            # METHOD 2: Median offset from all inliers (most robust)
            inlier_mask = mask.ravel().astype(bool)
            inlier_kp0 = kp0[inlier_mask]
            inlier_kp1 = kp1[inlier_mask]

            # Compute median offset from all inlier matches
            offsets_x = inlier_kp1[:, 0] - inlier_kp0[:, 0]
            offsets_y = inlier_kp1[:, 1] - inlier_kp0[:, 1]
            median_offset_x = np.median(offsets_x)
            median_offset_y = np.median(offsets_y)

            center_x, center_y = W0 / 2, H0 / 2
            xp_median = center_x + median_offset_x
            yp_median = center_y + median_offset_y

            print(f"    DEBUG: Drone processed shape (W,H) = ({W0}, {H0})")
            print(f"    DEBUG: Drone center = ({center_x:.1f}, {center_y:.1f})")
            print(f"    DEBUG: Median offset = ({median_offset_x:.1f}, {median_offset_y:.1f})")
            print(f"    DEBUG: Center method → ({xp_center:.1f}, {yp_center:.1f})")
            print(f"    DEBUG: Median method → ({xp_median:.1f}, {yp_median:.1f})")

            # Use median method (best result: ~19px error)
            xp, yp = xp_median, yp_median

            # Convert from processed crop coords to original crop coords
            x_in_crop, y_in_crop = proc_to_orig_xy(
                xp, yp,
                orig_w=map_crop.shape[1],  # 500
                orig_h=map_crop.shape[0],  # 500
                size=512,
                patch_size=16,
                square_ok=True
            )
            print(f"    DEBUG: Converted to original crop coords = ({x_in_crop:.1f}, {y_in_crop:.1f})")
            print(f"    DEBUG: Crop offset (x0, y0) = ({x0}, {y0})")

            # Convert to world coordinates
            rx, ry = x0 + x_in_crop, y0 + y_in_crop
            err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
            print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
                  f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

        # --- Save MASt3R match visualization if available ---
        if mast3r_result and mast3r_result[3] is not None:
            kp0, kp1_pts, _, H, mask, n_inliers, drone_proc_shape, crop_proc_shape = mast3r_result
            inlier_mask = mask.ravel().astype(bool)
            h_d, w_d = rgb_drone.shape[:2]
            h_c, w_c = rgb_crop.shape[:2]
            max_h = max(h_d, h_c)

            # Convert RGB back to BGR for visualization
            img_left = cv2.cvtColor(rgb_drone, cv2.COLOR_RGB2BGR)
            img_right = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)

            # Pad to match heights
            if h_d < max_h:
                img_left = cv2.copyMakeBorder(img_left, 0, max_h - h_d, 0, 0, cv2.BORDER_CONSTANT)
            if h_c < max_h:
                img_right = cv2.copyMakeBorder(img_right, 0, max_h - h_c, 0, 0, cv2.BORDER_CONSTANT)

            vis = np.hstack([img_left, img_right])

            # Draw inlier matches
            for i in range(len(kp0)):
                if inlier_mask[i]:
                    pt1 = (int(kp0[i, 0]), int(kp0[i, 1]))
                    pt2 = (int(kp1_pts[i, 0]) + w_d, int(kp1_pts[i, 1]))
                    cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
                    cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
                    cv2.circle(vis, pt2, 3, (0, 255, 0), -1)

            out = str(root / f"test_mast3r_{img_id}.png")
            cv2.imwrite(out, vis)
            print(f"    Saved MASt3R vis: {out}")
    else:
        print("\n  [!] MASt3R not available (requires: mast3r conda env + ~/mast3r folder)")

    # --- Save LoFTR match visualization if available ---
    if loftr_result and loftr_result[3] is not None:
        kp0, kp1_pts, conf, H, mask, n_inliers = loftr_result
        inlier_mask = mask.ravel().astype(bool)
        h_d, w_d = gray_drone_loftr.shape[:2]
        h_c, w_c = gray_crop.shape[:2]
        max_h = max(h_d, h_c)
        img_left = cv2.cvtColor(gray_drone_loftr, cv2.COLOR_GRAY2BGR)
        img_right = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        # Pad shorter image to match heights
        if h_d < max_h:
            img_left = cv2.copyMakeBorder(img_left, 0, max_h - h_d, 0, 0, cv2.BORDER_CONSTANT)
        if h_c < max_h:
            img_right = cv2.copyMakeBorder(img_right, 0, max_h - h_c, 0, 0, cv2.BORDER_CONSTANT)
        vis = np.hstack([img_left, img_right])
        for i in range(len(kp0)):
            if inlier_mask[i]:
                pt1 = (int(kp0[i, 0]), int(kp0[i, 1]))
                pt2 = (int(kp1_pts[i, 0]) + w_d, int(kp1_pts[i, 1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
                cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
                cv2.circle(vis, pt2, 3, (0, 255, 0), -1)
        out = str(root / f"test_loftr_{img_id}.png")
        cv2.imwrite(out, vis)
        print(f"\n  Saved LoFTR vis: {out}")

    # --- Save SIFT match visualization if we got matches ---
    best = sift_result or orb_result
    if best and best[3] is not None:
        kp1, kp2, good, H, mask, n_inliers = best
        inlier_matches = [good[i] for i in range(len(good)) if mask[i]]
        vis = cv2.drawMatches(
            drone_small, kp1, map_crop, kp2, inlier_matches, None,
            matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        out = str(root / f"test_match_{img_id}.png")
        cv2.imwrite(out, vis)
        print(f"\n  Saved match vis: {out}")


if __name__ == "__main__":
    main()
