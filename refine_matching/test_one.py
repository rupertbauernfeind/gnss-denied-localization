"""
Quick test: try multiple matching strategies on ONE image.

Usage:
    python test_one.py [image_id]
"""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import kornia.feature as KF

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


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    rough = load_csv(repo_root / "rough_matching" / "train_predictions.csv")
    gt = load_csv(repo_root / "data" / "train_data" / "train_pos.csv")
    image_dir = repo_root / "data" / "train_data" / "train_images"
    full_map = cv2.imread(str(repo_root / "data" / "map.png"))
    assert full_map is not None

    img_id = int(sys.argv[1]) if len(sys.argv) > 1 else sorted(set(rough) & set(gt))[0]

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
