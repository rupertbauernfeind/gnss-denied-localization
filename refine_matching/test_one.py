"""
Quick test: try multiple matching strategies on ONE image.

Usage:
    python test_one.py [image_id]
"""

import csv
import math
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

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

# Vismatch environment setup
try:
    from vismatch import get_matcher, available_models
    VISMATCH_AVAILABLE = True
except ImportError:
    VISMATCH_AVAILABLE = False

CROP_SIZE = 500
MAP_W, MAP_H = 5000, 2500


def read_rotation_from_xmp(xmp_path):
    """
    Read camera rotation matrix from .xmp file.
    Returns: 3x3 numpy rotation matrix, or None if not found.
    """
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Find the Rotation element
        # Namespace: http://www.capturingreality.com/ns/xcr/1.1#
        ns = {'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'}
        rotation_elem = root.find('.//xcr:Rotation', ns)

        if rotation_elem is not None and rotation_elem.text:
            # Parse 9 values (3x3 matrix flattened)
            values = list(map(float, rotation_elem.text.split()))
            if len(values) == 9:
                return np.array(values).reshape(3, 3)
    except Exception as e:
        print(f"    Warning: Could not read rotation from {xmp_path}: {e}")

    return None


def rotation_matrix_to_yaw(R):
    """
    Extract yaw angle (rotation around Z-axis) from 3x3 rotation matrix.
    Returns: yaw angle in degrees.
    """
    yaw_rad = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw_rad)


def rotate_image_with_matrix(image, R, compensate_roll=True):
    """
    Rotate image based on camera rotation matrix.

    Args:
        image: Input image (BGR or grayscale)
        R: 3x3 rotation matrix from camera
        compensate_roll: If True, rotate image to compensate for camera roll
                        (makes upside-down images right-side-up)

    Returns: Rotated image
    """
    if R is None:
        return image

    # Extract roll angle (rotation around viewing axis)
    # Roll is the rotation around the Z-axis (camera's forward direction)
    roll_rad = math.atan2(R[2, 1], R[2, 2])
    roll_deg = math.degrees(roll_rad)

    # For compensation, we want to rotate the image by -roll
    # (opposite of camera roll)
    rotation_angle = -roll_deg if compensate_roll else 0

    # Also compensate for yaw if map is north-aligned
    yaw_deg = rotation_matrix_to_yaw(R)
    rotation_angle -= yaw_deg

    # Rotate image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix (cv2 uses counterclockwise positive)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Calculate new image size to avoid cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust transformation matrix for new image size
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Apply rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0) if len(image.shape) == 3 else 0)

    return rotated


def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def save_match_visualization(output_dir, matcher_name, kp_drone, kp_crop, H, mask,
                              img_drone, img_crop, n_samples=20):
    """
    Save match data and visualization for a matcher.

    Args:
        output_dir: Directory to save files (pathlib.Path)
        matcher_name: Name of the matcher (e.g., "loftr", "mast3r")
        kp_drone: Keypoints in drone image (Nx2 numpy array)
        kp_crop: Keypoints in crop image (Nx2 numpy array)
        H: Homography matrix (3x3 numpy array)
        mask: Inlier mask (Nx1 numpy array)
        img_drone: Drone image (BGR or grayscale)
        img_crop: Map crop image (BGR or grayscale)
        n_samples: Number of inlier matches to visualize
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert mask to boolean array
    if mask is not None:
        inlier_mask = mask.ravel().astype(bool)
        inlier_indices = np.where(inlier_mask)[0]
        n_inliers = len(inlier_indices)
    else:
        n_inliers = 0
        inlier_indices = np.array([])

    # Save match data
    np.savez(
        str(output_dir / f"{matcher_name}_matches.npz"),
        kp_drone=kp_drone,
        kp_crop=kp_crop,
        H=H,
        mask=mask,
        inlier_indices=inlier_indices,
        n_matches=len(kp_drone),
        n_inliers=n_inliers
    )

    # Create visualization with sampled inlier matches
    if n_inliers == 0:
        print(f"    Warning: No inliers to visualize for {matcher_name}")
        return

    n_sample = min(n_samples, n_inliers)
    sample_indices = np.random.choice(inlier_indices, n_sample, replace=False)

    # Convert grayscale to BGR if needed
    if len(img_drone.shape) == 2:
        img_drone = cv2.cvtColor(img_drone, cv2.COLOR_GRAY2BGR)
    if len(img_crop.shape) == 2:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)

    # Create side-by-side visualization
    h_d, w_d = img_drone.shape[:2]
    h_c, w_c = img_crop.shape[:2]
    max_h = max(h_d, h_c)

    # Pad to match heights
    img_left = img_drone.copy()
    img_right = img_crop.copy()

    if h_d < max_h:
        img_left = cv2.copyMakeBorder(img_left, 0, max_h - h_d, 0, 0, cv2.BORDER_CONSTANT)
    if h_c < max_h:
        img_right = cv2.copyMakeBorder(img_right, 0, max_h - h_c, 0, 0, cv2.BORDER_CONSTANT)

    vis = np.hstack([img_left, img_right])

    # Draw sampled matches with random colors
    for i in sample_indices:
        x1 = int(kp_drone[i, 0])
        y1 = int(kp_drone[i, 1])
        x2 = int(kp_crop[i, 0])
        y2 = int(kp_crop[i, 1])

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
    text = f"{matcher_name}: {n_sample}/{n_inliers} inliers shown"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save visualization
    cv2.imwrite(str(output_dir / f"{matcher_name}_visualization.png"), vis)
    print(f"    Saved {matcher_name} data and visualization")


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
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
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
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
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
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
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
        images = load_images([drone_path, crop_path], size=512)

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

        if len(matches_im0) < 4:
            return None

        # Compute homography (matches are already numpy arrays)
        src = matches_im0.reshape(-1, 1, 2).astype(np.float32)
        dst = matches_im1.reshape(-1, 1, 2).astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        n_inliers = int(mask.sum()) if mask is not None else 0
        print(f"    RANSAC inliers: {n_inliers}")

        if n_inliers < 4:
            return None

        return matches_im0, matches_im1, None, H, mask, n_inliers
    finally:
        # Clean up temporary files
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)


def square_pad_resize(image, target_size):
    """
    Resize image to square by padding shorter dimension.

    Returns:
        padded: Square image of size target_size × target_size
        pad_offset: (pad_w, pad_h) offset of original content
        resized_size: (w, h) size before padding
    """
    h, w = image.shape[:2]

    # Resize keeping aspect ratio
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_size - new_h - pad_h,
            pad_w, target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, target_size - new_h - pad_h,
            pad_w, target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return padded, (pad_w, pad_h), (new_w, new_h)


def try_vismatch(gray_drone, gray_crop, matcher_name, device='cpu'):
    """Vismatch universal matcher. Returns (kp0, kp1, None, H, mask, n_inliers)."""
    import tempfile
    import os
    from PIL import Image

    # Get matcher
    matcher = get_matcher(matcher_name, device=device, max_num_keypoints=4096)

    # Convert grayscale to RGB (vismatch expects RGB)
    # Create 3-channel image from grayscale
    drone_rgb = cv2.cvtColor(gray_drone, cv2.COLOR_GRAY2RGB)
    crop_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)

    # Pre-resize both to 512×512 square (fixing Bug #1: asymmetric scaling)
    drone_square, drone_pad_offset, drone_resized_size = square_pad_resize(drone_rgb, 512)
    crop_square, crop_pad_offset, crop_resized_size = square_pad_resize(crop_rgb, 512)

    # Convert to PIL and save to temp files (vismatch load_image expects file paths)
    drone_pil = Image.fromarray(drone_square)
    crop_pil = Image.fromarray(crop_square)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_drone:
        drone_pil.save(tmp_drone.name, 'JPEG')
        drone_path = tmp_drone.name

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_crop:
        crop_pil.save(tmp_crop.name, 'JPEG')
        crop_path = tmp_crop.name

    try:
        # Load WITHOUT vismatch's internal resize (we already resized to 512×512)
        img0 = matcher.load_image(drone_path, resize=None)
        img1 = matcher.load_image(crop_path, resize=None)

        # Match
        with torch.inference_mode():
            result = matcher(img0, img1)

        n_inliers = result['num_inliers']
        H = result['H']
        matched_kpts0 = result.get('matched_kpts0', [])
        matched_kpts1 = result.get('matched_kpts1', [])
        inlier_mask = result.get('inlier_mask', None)

        print(f"    Vismatch ({matcher_name}): {len(matched_kpts0)} matches, {n_inliers} inliers")

        if n_inliers < 4 or H is None:
            return None

        # H maps from 512×512 padded space to 512×512 padded space
        # Need to transform to gray_drone coords → gray_crop coords

        h_orig_drone, w_orig_drone = gray_drone.shape[:2]
        h_orig_crop, w_orig_crop = gray_crop.shape[:2]

        # Step 1: Remove padding offset from H (shift coordinates back to resized space)
        # Translation to account for padding: points in padded space = points in resized space + offset
        drone_pad_w, drone_pad_h = drone_pad_offset
        crop_pad_w, crop_pad_h = crop_pad_offset

        # Create translation matrices to remove padding
        T_drone_inv = np.array([
            [1, 0, -drone_pad_w],
            [0, 1, -drone_pad_h],
            [0, 0, 1]
        ], dtype=np.float64)

        T_crop = np.array([
            [1, 0, crop_pad_w],
            [0, 1, crop_pad_h],
            [0, 0, 1]
        ], dtype=np.float64)

        # H_resized maps from resized drone to resized crop (no padding)
        H_resized = T_crop @ H @ T_drone_inv

        # Step 2: Scale from resized space to original space
        drone_w_resized, drone_h_resized = drone_resized_size
        crop_w_resized, crop_h_resized = crop_resized_size

        # Scale drone: original -> resized (divide by scale factor)
        S_drone = np.diag([
            drone_w_resized / w_orig_drone,
            drone_h_resized / h_orig_drone,
            1.0
        ])

        # Scale crop: resized -> original (multiply by scale factor)
        S_crop_inv = np.diag([
            w_orig_crop / crop_w_resized,
            h_orig_crop / crop_h_resized,
            1.0
        ])

        # Final homography: original drone coords → original crop coords
        H_scaled = S_crop_inv @ H_resized @ S_drone

        # Scale matched keypoints back to original coordinates
        if isinstance(matched_kpts0, torch.Tensor):
            matched_kpts0 = matched_kpts0.cpu().numpy()
        if isinstance(matched_kpts1, torch.Tensor):
            matched_kpts1 = matched_kpts1.cpu().numpy()
        if isinstance(inlier_mask, torch.Tensor):
            inlier_mask = inlier_mask.cpu().numpy()

        # Transform keypoints: padded 512×512 → resized → original
        if len(matched_kpts0) > 0:
            # Step 1: Remove padding offset
            kpts0_nopad = matched_kpts0.copy()
            kpts1_nopad = matched_kpts1.copy()
            kpts0_nopad[:, 0] -= drone_pad_w
            kpts0_nopad[:, 1] -= drone_pad_h
            kpts1_nopad[:, 0] -= crop_pad_w
            kpts1_nopad[:, 1] -= crop_pad_h

            # Step 2: Scale to original coordinates
            kpts0_scaled = kpts0_nopad.copy()
            kpts1_scaled = kpts1_nopad.copy()
            kpts0_scaled[:, 0] *= w_orig_drone / drone_w_resized
            kpts0_scaled[:, 1] *= h_orig_drone / drone_h_resized
            kpts1_scaled[:, 0] *= w_orig_crop / crop_w_resized
            kpts1_scaled[:, 1] *= h_orig_crop / crop_h_resized
        else:
            kpts0_scaled = np.array([])
            kpts1_scaled = np.array([])

        # Create mask in the expected format (Nx1)
        # If inlier_mask not provided, compute it from homography with stricter threshold (Phase 4)
        if inlier_mask is not None:
            mask = inlier_mask.reshape(-1, 1)
        elif len(kpts0_scaled) > 0 and H_scaled is not None:
            # Compute inlier mask by checking reprojection error
            src_pts = kpts0_scaled.reshape(-1, 1, 2).astype(np.float32)
            dst_pts = kpts1_scaled.reshape(-1, 1, 2).astype(np.float32)
            # Use stricter RANSAC threshold (2.0 instead of 5.0) - Phase 4
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        else:
            mask = None

        return kpts0_scaled, kpts1_scaled, None, H_scaled, mask, n_inliers
    finally:
        # Clean up temporary files
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)


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

    # Create output directory for match visualizations
    output_dir = root / "match_visualizations" / f"img_{img_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {output_dir}")

    # Crop map (preserve sub-pixel precision for position calculation)
    cx_float, cy_float = rough_x, rough_y
    x0_float = max(0.0, min(float(MAP_W - CROP_SIZE), cx_float - CROP_SIZE / 2.0))
    y0_float = max(0.0, min(float(MAP_H - CROP_SIZE), cy_float - CROP_SIZE / 2.0))

    # Round only for array indexing
    x0, y0 = int(round(x0_float)), int(round(y0_float))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]

    # Load drone image
    drone = cv2.imread(str(image_dir / f"{img_id:04d}.JPG"))
    assert drone is not None
    h_orig, w_orig = drone.shape[:2]
    print(f"  Drone image: {w_orig}x{h_orig}")

    # Load and apply rotation correction from .xmp file
    xmp_path = repo_root / "rough_matching" / "realityscan_positions" / "train" / f"{img_id:04d}.xmp"
    rotation_matrix = read_rotation_from_xmp(xmp_path)

    if rotation_matrix is not None:
        roll_deg = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw_deg = rotation_matrix_to_yaw(rotation_matrix)
        print(f"  Camera rotation: yaw={yaw_deg:.1f}°, roll={roll_deg:.1f}°")
        print(f"  Applying rotation correction...")

        # Rotate drone image to align with map
        drone = rotate_image_with_matrix(drone, rotation_matrix, compensate_roll=True)
        h_orig, w_orig = drone.shape[:2]
        print(f"  Rotated drone image: {w_orig}x{h_orig}")
    else:
        print(f"  Warning: No rotation data found, proceeding without correction")

    # Resize drone for feature matching
    resize_w = 800
    scale_f = resize_w / w_orig
    drone_small = cv2.resize(drone, (resize_w, int(h_orig * scale_f)))

    gray_crop = cv2.cvtColor(map_crop, cv2.COLOR_BGR2GRAY)
    gray_drone = cv2.cvtColor(drone_small, cv2.COLOR_BGR2GRAY)
    gray_drone_full = cv2.cvtColor(drone, cv2.COLOR_BGR2GRAY)

    # Save input images for reference
    cv2.imwrite(str(output_dir / "drone_resized.jpg"), drone_small)
    cv2.imwrite(str(output_dir / "map_crop.jpg"), map_crop)
    with open(output_dir / "info.txt", "w") as f:
        f.write(f"Image: {img_id}\n")
        f.write(f"GT: ({gt_x:.1f}, {gt_y:.1f})\n")
        f.write(f"Rough: ({rough_x:.1f}, {rough_y:.1f})\n")
        f.write(f"Rough error: {rough_err:.1f}px ({rough_err/5:.0f}m)\n")
        f.write(f"Crop offset: ({x0}, {y0})\n")
        f.write(f"Drone size: {w_orig}x{h_orig} (original), {drone_small.shape[1]}x{drone_small.shape[0]} (resized)\n")

    # --- Strategy 1: SIFT (permissive ratio) ---
    print("\n  [1] SIFT (ratio=0.85, low contrast threshold):")
    sift_result = try_sift(gray_drone, gray_crop, ratio=0.85)

    if sift_result and sift_result[5] >= 6:
        kp1, kp2, good, H, mask, n_inliers = sift_result

        # Extract keypoints for visualization and median calculation
        kp_drone = np.float32([kp1[m.queryIdx].pt for m in good])
        kp_crop = np.float32([kp2[m.trainIdx].pt for m in good])

        # Phase 2: Use median of transformed inliers instead of center projection
        inlier_mask = mask.ravel().astype(bool)
        inlier_kp_drone = kp_drone[inlier_mask]

        if len(inlier_kp_drone) > 0:
            # Transform all inlier keypoints
            transformed = cv2.perspectiveTransform(
                inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
            ).reshape(-1, 2)

            # Median is more robust to outliers
            median_x = np.median(transformed[:, 0])
            median_y = np.median(transformed[:, 1])

            rx, ry = median_x + x0_float, median_y + y0_float
        else:
            # Fallback to center
            h_s, w_s = gray_drone.shape[:2]
            center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(center, H)
            rx, ry = proj[0, 0, 0] + x0_float, proj[0, 0, 1] + y0_float

        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

        # Save matches and visualization
        save_match_visualization(output_dir, "sift", kp_drone, kp_crop, H, mask,
                                 gray_drone, gray_crop)

    # --- Strategy 2: ORB ---
    print("\n  [2] ORB:")
    orb_result = try_orb(gray_drone, gray_crop)

    if orb_result and orb_result[5] >= 6:
        kp1, kp2, good, H, mask, n_inliers = orb_result

        # Extract keypoints for visualization and median calculation
        kp_drone = np.float32([kp1[m.queryIdx].pt for m in good])
        kp_crop = np.float32([kp2[m.trainIdx].pt for m in good])

        # Phase 2: Use median of transformed inliers instead of center projection
        inlier_mask = mask.ravel().astype(bool)
        inlier_kp_drone = kp_drone[inlier_mask]

        if len(inlier_kp_drone) > 0:
            # Transform all inlier keypoints
            transformed = cv2.perspectiveTransform(
                inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
            ).reshape(-1, 2)

            # Median is more robust to outliers
            median_x = np.median(transformed[:, 0])
            median_y = np.median(transformed[:, 1])

            rx, ry = median_x + x0_float, median_y + y0_float
        else:
            # Fallback to center
            h_s, w_s = gray_drone.shape[:2]
            center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(center, H)
            rx, ry = proj[0, 0, 0] + x0_float, proj[0, 0, 1] + y0_float

        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

        # Save matches and visualization
        save_match_visualization(output_dir, "orb", kp_drone, kp_crop, H, mask,
                                 gray_drone, gray_crop)

    # --- Strategy 3: Multi-scale template matching ---
    print("\n  [3] Template matching (multi-scale):")
    tmpl_result = try_template_multiscale(gray_drone_full, gray_crop)

    if tmpl_result:
        crop_cx, crop_cy, score, best_scale = tmpl_result
        rx, ry = crop_cx + x0_float, crop_cy + y0_float
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
        kp_drone, kp_crop, conf, H, mask, n_inliers = loftr_result

        # Phase 2: Use median of transformed inliers instead of center projection
        inlier_mask = mask.ravel().astype(bool)
        inlier_kp_drone = kp_drone[inlier_mask]

        if len(inlier_kp_drone) > 0:
            # Transform all inlier keypoints
            transformed = cv2.perspectiveTransform(
                inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
            ).reshape(-1, 2)

            # Median is more robust to outliers
            median_x = np.median(transformed[:, 0])
            median_y = np.median(transformed[:, 1])

            rx, ry = median_x + x0, median_y + y0
        else:
            # Fallback to center
            h_s, w_s = gray_drone_loftr.shape[:2]
            center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(center, H)
            rx, ry = proj[0, 0, 0] + x0, proj[0, 0, 1] + y0

        err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
        print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
              f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

        # Save matches and visualization
        save_match_visualization(output_dir, "loftr", kp_drone, kp_crop, H, mask,
                                 gray_drone_loftr, gray_crop)

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
            kp_drone, kp_crop, _, H, mask, n_inliers = mast3r_result

            # Phase 2: Use median of transformed inliers instead of center projection
            inlier_mask = mask.ravel().astype(bool)
            inlier_kp_drone = kp_drone[inlier_mask]

            if len(inlier_kp_drone) > 0:
                # Transform all inlier keypoints
                transformed = cv2.perspectiveTransform(
                    inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
                ).reshape(-1, 2)

                # Median is more robust to outliers
                median_x = np.median(transformed[:, 0])
                median_y = np.median(transformed[:, 1])

                rx, ry = median_x + x0_float, median_y + y0_float
            else:
                # Fallback to center
                h_s, w_s = drone_mast3r.shape[:2]
                center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(center, H)
                rx, ry = proj[0, 0, 0] + x0_float, proj[0, 0, 1] + y0_float
            err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
            print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
                  f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

            # Save matches and visualization
            save_match_visualization(output_dir, "mast3r", kp_drone, kp_crop, H, mask,
                                     drone_mast3r, map_crop)
    else:
        print("\n  [!] MASt3R not available (requires: mast3r conda env + ~/mast3r folder)")

    # --- Strategy 6+: Vismatch matchers ---
    if VISMATCH_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test XoFTR and master
        vismatch_matchers = [
            'xoftr',                 # XoFTR detector-free matcher
            'master',                # master (possibly related to MASt3R)
            # 'omniglue',              # State-of-the-art (requires TensorFlow)
            # 'roma',                  # Dense detector-free matcher
            # 'superpoint-lightglue',  # Fast, accurate sparse matcher
            # 'xfeat',                 # Efficient learned features
            # 'dedode-lightglue',      # DeDoDe detector + LightGlue
        ]

        for idx, matcher_name in enumerate(vismatch_matchers, start=6):
            print(f"\n  [{idx}] Vismatch - {matcher_name}:")

            try:
                # Use gray_drone (resized to 800px) and gray_crop (500x500)
                vismatch_result = try_vismatch(gray_drone, gray_crop, matcher_name, device=device)

                if vismatch_result and vismatch_result[5] >= 4:
                    kp_drone, kp_crop, _, H, mask, n_inliers = vismatch_result

                    # Use median displacement method (more robust than homography transform)
                    inlier_mask = mask.ravel().astype(bool)
                    inlier_kp_drone = kp_drone[inlier_mask]
                    inlier_kp_crop = kp_crop[inlier_mask]

                    if len(inlier_kp_drone) > 0:
                        # Calculate displacement for each match
                        displacements = inlier_kp_crop - inlier_kp_drone

                        # Take median displacement
                        median_dx = np.median(displacements[:, 0])
                        median_dy = np.median(displacements[:, 1])

                        # Apply to center of drone image
                        h_s, w_s = gray_drone.shape[:2]
                        center_x = w_s / 2.0
                        center_y = h_s / 2.0

                        crop_x = center_x + median_dx
                        crop_y = center_y + median_dy

                        rx, ry = crop_x + x0_float, crop_y + y0_float
                    else:
                        # Fallback to center if no inliers (shouldn't happen with n_inliers >= 4)
                        h_s, w_s = gray_drone.shape[:2]
                        center = np.float32([[w_s / 2, h_s / 2]]).reshape(-1, 1, 2)
                        proj = cv2.perspectiveTransform(center, H)
                        rx, ry = proj[0, 0, 0] + x0_float, proj[0, 0, 1] + y0_float

                    err = math.sqrt((rx - gt_x) ** 2 + (ry - gt_y) ** 2)
                    print(f"    -> Refined: ({rx:.1f}, {ry:.1f})  err={err:.1f}px ({err/5:.0f}m)  "
                          f"{'BETTER' if err < rough_err else 'WORSE'} (delta={rough_err-err:+.1f}px)")

                    # Save matches and visualization
                    if kp_drone is not None and len(kp_drone) > 0:
                        save_match_visualization(output_dir, matcher_name, kp_drone, kp_crop, H, mask,
                                                 gray_drone, gray_crop)
                else:
                    print(f"    -> Failed (insufficient matches)")
            except Exception as e:
                print(f"    -> Error: {e}")
    else:
        print("\n  [!] Vismatch not available")

    print(f"\n  All match data saved to: {output_dir}")


if __name__ == "__main__":
    main()
