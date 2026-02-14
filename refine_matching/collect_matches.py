#!/usr/bin/env python3
"""
Batch collect matches for multiple images to enable rapid position calculation tuning.

This script runs all matchers on a range of images and saves the match data,
allowing analyze_matches.py to quickly test different position calculation methods.
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
from tqdm import tqdm

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

CROP_SIZE = 750
MAP_W, MAP_H = 5000, 2500


def load_csv(path):
    """Load CSV file into dict mapping image_id -> (x, y)."""
    result = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different CSV formats
            if "image_id" in row:
                img_id = int(row["image_id"])
                x, y = float(row["x"]), float(row["y"])
            elif "id" in row:
                img_id = int(row["id"])
                x, y = float(row["x_pixel"]), float(row["y_pixel"])
            else:
                continue
            result[img_id] = (x, y)
    return result


def read_rotation_from_xmp(xmp_path):
    """Read camera rotation matrix from .xmp file."""
    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Define namespaces
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'
        }

        # Find the Rotation element
        rotation_elem = root.find('.//xcr:Rotation', namespaces)
        if rotation_elem is not None:
            # Parse the 9 values (row-major 3x3 matrix)
            values = list(map(float, rotation_elem.text.split()))
            if len(values) == 9:
                return np.array(values).reshape(3, 3)
    except Exception as e:
        print(f"Warning: Could not read rotation from {xmp_path}: {e}")
    return None


def rotation_matrix_to_yaw(R):
    """Extract yaw angle in degrees from rotation matrix."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw)


def rotate_image_with_matrix(image, R, compensate_roll=True):
    """Rotate image to align with map using camera rotation matrix."""
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    if compensate_roll:
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        roll = 0.0

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Rotation angle: yaw + roll (combine to align with map)
    angle = yaw + roll

    # Create rotation matrix and rotate
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image size to prevent cropping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
    return rotated


def square_pad_resize(image, target_size):
    """Resize image to square by padding shorter dimension."""
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
        padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
                                     pad_w, target_size - new_w - pad_w,
                                     cv2.BORDER_CONSTANT, value=0)
    else:
        padded = cv2.copyMakeBorder(resized, pad_h, target_size - new_h - pad_h,
                                     pad_w, target_size - new_w - pad_w,
                                     cv2.BORDER_CONSTANT, value=(0,0,0))

    return padded, (pad_w, pad_h), (new_w, new_h)


def try_sift(gray_drone, gray_crop, ratio=0.8):
    """Run SIFT matcher."""
    sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=8)

    kp1, des1 = sift.detectAndCompute(gray_drone, None)
    kp2, des2 = sift.detectAndCompute(gray_crop, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 6:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if H is None:
        return None

    n_inliers = int(mask.sum())

    # Return keypoints as arrays for saving
    kp_drone = np.float32([kp1[m.queryIdx].pt for m in good])
    kp_crop = np.float32([kp2[m.trainIdx].pt for m in good])

    return kp_drone, kp_crop, good, H, mask, n_inliers


def try_orb(gray_drone, gray_crop):
    """Run ORB matcher."""
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=12)

    kp1, des1 = orb.detectAndCompute(gray_drone, None)
    kp2, des2 = orb.detectAndCompute(gray_crop, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 6:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if H is None:
        return None

    n_inliers = int(mask.sum())

    kp_drone = np.float32([kp1[m.queryIdx].pt for m in good])
    kp_crop = np.float32([kp2[m.trainIdx].pt for m in good])

    return kp_drone, kp_crop, good, H, mask, n_inliers


def try_loftr(gray_drone, gray_crop, conf_thresh=0.5):
    """Run LoFTR matcher."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    matcher = KF.LoFTR(pretrained='outdoor')
    matcher.to(device)
    matcher.eval()

    img0 = torch.from_numpy(gray_drone).float()[None, None] / 255.0
    img1 = torch.from_numpy(gray_crop).float()[None, None] / 255.0
    img0 = img0.to(device)
    img1 = img1.to(device)

    with torch.inference_mode():
        correspondences = matcher({'image0': img0, 'image1': img1})

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()

    mask_conf = conf > conf_thresh
    mkpts0 = mkpts0[mask_conf]
    mkpts1 = mkpts1[mask_conf]

    if len(mkpts0) < 6:
        return None

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 2.0)

    if H is None:
        return None

    n_inliers = int(mask.sum())

    return mkpts0, mkpts1, None, H, mask, n_inliers


def try_vismatch(gray_drone, gray_crop, matcher_name, device='cpu'):
    """Run vismatch matcher (XoFTR)."""
    if not VISMATCH_AVAILABLE:
        return None

    import tempfile

    # Pre-resize both images to 512×512 with padding
    drone_padded, drone_pad_offset, drone_resized_size = square_pad_resize(gray_drone, 512)
    crop_padded, crop_pad_offset, crop_resized_size = square_pad_resize(gray_crop, 512)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_drone, \
         tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_crop:

        drone_path = f_drone.name
        crop_path = f_crop.name

        cv2.imwrite(drone_path, drone_padded)
        cv2.imwrite(crop_path, crop_padded)

        try:
            matcher = get_matcher(matcher_name, device=device)

            # Load WITHOUT vismatch's internal resizing (already 512×512)
            img0 = matcher.load_image(drone_path, resize=None)
            img1 = matcher.load_image(crop_path, resize=None)

            result = matcher(img0, img1)

            n_inliers = result['num_inliers']
            H = result['H']

            # Different matchers return different keys
            # XoFTR: 'kpts0', 'kpts1'
            # Master: 'matched_kpts0', 'matched_kpts1'
            kpts0 = result.get('kpts0', result.get('matched_kpts0', np.array([])))
            kpts1 = result.get('kpts1', result.get('matched_kpts1', np.array([])))

            if isinstance(kpts0, torch.Tensor):
                kpts0 = kpts0.cpu().numpy()
            if isinstance(kpts1, torch.Tensor):
                kpts1 = kpts1.cpu().numpy()

            if len(kpts0) < 6:
                return None

            # Remove padding offset from keypoints
            kpts0_unpadded = kpts0 - np.array(drone_pad_offset)
            kpts1_unpadded = kpts1 - np.array(crop_pad_offset)

            # Scale to original gray_drone and gray_crop coordinates
            h_drone, w_drone = gray_drone.shape
            h_crop, w_crop = gray_crop.shape

            scale_drone_x = w_drone / drone_resized_size[0]
            scale_drone_y = h_drone / drone_resized_size[1]
            scale_crop_x = w_crop / crop_resized_size[0]
            scale_crop_y = h_crop / crop_resized_size[1]

            kpts0_scaled = kpts0_unpadded * np.array([scale_drone_x, scale_drone_y])
            kpts1_scaled = kpts1_unpadded * np.array([scale_crop_x, scale_crop_y])

            # Recompute homography in original coordinate space
            src_pts = kpts0_scaled.reshape(-1, 1, 2).astype(np.float32)
            dst_pts = kpts1_scaled.reshape(-1, 1, 2).astype(np.float32)
            H_scaled, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

            if H_scaled is None:
                return None

            n_inliers = int(mask.sum())

            return kpts0_scaled, kpts1_scaled, None, H_scaled, mask, n_inliers

        finally:
            import os
            os.unlink(drone_path)
            os.unlink(crop_path)


def try_mast3r(gray_drone, gray_crop):
    """Run MASt3R matcher."""
    if not MAST3R_AVAILABLE:
        return None

    import tempfile

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)

    # Convert grayscale to RGB
    rgb_drone = cv2.cvtColor(gray_drone, cv2.COLOR_GRAY2RGB)
    rgb_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_drone, \
         tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_crop:

        drone_path = f_drone.name
        crop_path = f_crop.name

        cv2.imwrite(drone_path, rgb_drone)
        cv2.imwrite(crop_path, rgb_crop)

        try:
            images = load_images([drone_path, crop_path], size=512)
            output = inference([tuple(images)], model, device, batch_size=1)

            # Extract matches
            view1, view2 = output['view1'], output['view2']
            pts3d_1 = view1['pts3d']
            pts3d_2 = view2['pts3d_in_other_view']

            # Find reciprocal nearest neighbors
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                pts3d_1.cpu().numpy(), pts3d_2.cpu().numpy(),
                subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13
            )

            H, W, _ = pts3d_1.shape
            grid_y, grid_x = np.mgrid[:H, :W]

            # Get matched keypoints
            mkpts0 = np.stack([grid_x[matches_im0[:, 0], matches_im0[:, 1]],
                               grid_y[matches_im0[:, 0], matches_im0[:, 1]]], axis=1)
            mkpts1 = np.stack([grid_x[matches_im1[:, 0], matches_im1[:, 1]],
                               grid_y[matches_im1[:, 0], matches_im1[:, 1]]], axis=1)

            # Scale to original image coordinates
            h_drone, w_drone = gray_drone.shape
            h_crop, w_crop = gray_crop.shape

            mkpts0_scaled = mkpts0 * np.array([w_drone / W, h_drone / H])
            mkpts1_scaled = mkpts1 * np.array([w_crop / W, h_crop / H])

            if len(mkpts0_scaled) < 6:
                return None

            # Compute homography
            H_mat, mask = cv2.findHomography(mkpts0_scaled, mkpts1_scaled, cv2.RANSAC, 2.0)

            if H_mat is None:
                return None

            n_inliers = int(mask.sum())

            return mkpts0_scaled, mkpts1_scaled, None, H_mat, mask, n_inliers

        finally:
            import os
            os.unlink(drone_path)
            os.unlink(crop_path)


def visualize_matches(drone_img, map_crop, kp_drone, kp_crop, mask, matcher_name, n_inliers):
    """Create a side-by-side visualization with match lines."""
    h_d, w_d = drone_img.shape[:2]
    h_c, w_c = map_crop.shape[:2]

    h_max = max(h_d, h_c)

    vis = np.zeros((h_max, w_d + w_c, 3), dtype=np.uint8)
    vis[:h_d, :w_d] = drone_img
    vis[:h_c, w_d:] = map_crop

    # Draw match lines for inliers
    if mask is not None and len(kp_drone) > 0:
        inlier_mask = mask.ravel().astype(bool)
        inlier_kp_drone = kp_drone[inlier_mask]
        inlier_kp_crop = kp_crop[inlier_mask]

        # Sample matches to avoid overcrowding (max 100 lines)
        n_to_draw = min(100, len(inlier_kp_drone))
        indices = np.random.choice(len(inlier_kp_drone), n_to_draw, replace=False)

        for idx in indices:
            pt1 = tuple(inlier_kp_drone[idx].astype(int))
            pt2 = tuple((inlier_kp_crop[idx] + np.array([w_d, 0])).astype(int))

            # Draw line
            color = (0, 255, 0)  # Green for inliers
            cv2.line(vis, pt1, pt2, color, 1)

            # Draw circles at keypoints
            cv2.circle(vis, pt1, 2, (0, 255, 255), -1)  # Yellow on drone
            cv2.circle(vis, pt2, 2, (0, 255, 255), -1)  # Yellow on crop

    # Add text
    cv2.putText(vis, f"{matcher_name}: {n_inliers} inliers",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return vis


def collect_matches_for_image(img_id, root_dir, rough_dict, gt_dict, full_map, image_type='train'):
    """Collect matches from all matchers for one image.

    Args:
        img_id: Image ID number
        root_dir: Root directory path
        rough_dict: Dictionary of rough GPS predictions
        gt_dict: Dictionary of ground truth positions (can be None for test images)
        full_map: Full satellite map image
        image_type: 'train' or 'test'
    """
    print(f"\n{'='*80}")
    print(f"Processing Image {img_id:04d} ({image_type.upper()})")
    print(f"{'='*80}")

    # Check if we have rough GPS for this image
    if img_id not in rough_dict:
        print(f"  ✗ Skipping image {img_id:04d}: missing rough GPS")
        return False

    rough_x, rough_y = rough_dict[img_id]

    # Ground truth is optional (only available for train images)
    has_gt = gt_dict is not None and img_id in gt_dict
    if has_gt:
        gt_x, gt_y = gt_dict[img_id]
        rough_err = math.sqrt((rough_x - gt_x)**2 + (rough_y - gt_y)**2)
        print(f"Ground Truth: ({gt_x:.1f}, {gt_y:.1f})")
        print(f"Rough GPS:    ({rough_x:.1f}, {rough_y:.1f}) - Error: {rough_err:.1f}px")
    else:
        print(f"Rough GPS:    ({rough_x:.1f}, {rough_y:.1f})")

    # Load drone image
    image_dir = root_dir / "data" / f"{image_type}_data" / f"{image_type}_images"
    drone_img = cv2.imread(str(image_dir / f"{img_id:04d}.JPG"))
    if drone_img is None:
        print(f"  ✗ Skipping image {img_id:04d}: failed to load drone image")
        return False

    h_orig, w_orig = drone_img.shape[:2]
    print(f"  Drone image: {w_orig}x{h_orig}")

    # Load and apply rotation correction
    xmp_path = root_dir / "rough_matching" / "realityscan_positions" / image_type / f"{img_id:04d}.xmp"
    rotation_matrix = read_rotation_from_xmp(xmp_path)

    if rotation_matrix is not None:
        roll_deg = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw_deg = rotation_matrix_to_yaw(rotation_matrix)
        print(f"  Camera rotation: yaw={yaw_deg:.1f}°, roll={roll_deg:.1f}°")
        print(f"  Applying rotation correction...")

        drone_img = rotate_image_with_matrix(drone_img, rotation_matrix, compensate_roll=True)
        h_orig, w_orig = drone_img.shape[:2]
        print(f"  Rotated drone image: {w_orig}x{h_orig}")
    else:
        print(f"  Warning: No rotation data found, proceeding without correction")

    # Resize drone for feature matching
    resize_w = 800
    scale_f = resize_w / w_orig
    drone_small = cv2.resize(drone_img, (resize_w, int(h_orig * scale_f)))

    gray_drone = cv2.cvtColor(drone_small, cv2.COLOR_BGR2GRAY)

    # Crop extraction with float precision
    cx_float, cy_float = rough_x, rough_y
    x0_float = max(0.0, min(float(MAP_W - CROP_SIZE), cx_float - CROP_SIZE / 2.0))
    y0_float = max(0.0, min(float(MAP_H - CROP_SIZE), cy_float - CROP_SIZE / 2.0))

    x0, y0 = int(round(x0_float)), int(round(y0_float))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]
    gray_crop = cv2.cvtColor(map_crop, cv2.COLOR_BGR2GRAY)

    print(f"Crop offset: ({x0}, {y0})")

    # Create output directory for this image
    output_dir = Path(__file__).parent / "match_visualizations" / f"img_{img_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try Master (MASt3R) via vismatch only
    matchers_results = []

    print("\n--- Master (via vismatch) ---")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = try_vismatch(gray_drone, gray_crop, 'master', device=device)
        if result is not None:
            kp_drone, kp_crop, kp_viz, H, mask, n_inliers = result
            print(f"  ✓ Master: {n_inliers} inliers")

            # Prepare NPZ data
            npz_data = {
                'kp_drone': kp_drone,
                'kp_crop': kp_crop,
                'H': H,
                'mask': mask,
                'n_inliers': n_inliers,
                'crop_offset': np.array([x0, y0]),
                'drone_shape': gray_drone.shape,
                'rough_position': np.array([rough_x, rough_y])
            }
            # Add ground truth only if available
            if has_gt:
                npz_data['gt_position'] = np.array([gt_x, gt_y])

            np.savez(output_dir / "master_matches.npz", **npz_data)
            vis_img = visualize_matches(drone_small, map_crop, kp_drone, kp_crop, mask, "Master", n_inliers)
            cv2.imwrite(str(output_dir / "master_visualization.png"), vis_img)
            matchers_results.append(("Master", n_inliers))
        else:
            print(f"  ✗ Master failed")
    except Exception as e:
        print(f"  ✗ Master error: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print(f"Image {img_id:04d} Summary:")
    if has_gt:
        print(f"  Ground Truth: ({gt_x:.1f}, {gt_y:.1f})")
        print(f"  Rough GPS Error: {rough_err:.1f}px")
    else:
        print(f"  Rough GPS: ({rough_x:.1f}, {rough_y:.1f}) [No ground truth]")
    print(f"  Master (via vismatch): {matchers_results[0][1] if matchers_results else 0} inliers")
    print(f"{'='*80}")

    return len(matchers_results) > 0


def main():
    """Collect matches for all train and test images."""
    print("="*80)
    print("BATCH MATCH COLLECTION - ALL TRAIN & TEST IMAGES")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE} pixels")
    print("="*80)

    # Load data once
    root = Path(__file__).parent.parent

    # Load rough predictions for both train and test
    train_rough_dict = load_csv(root / "rough_matching" / "train_predictions.csv")
    test_rough_dict = load_csv(root / "rough_matching" / "test_predicted.csv")

    # Load ground truth (only available for train)
    gt_dict = load_csv(root / "data" / "train_data" / "train_pos.csv")

    # Load map
    full_map = cv2.imread(str(root / "data" / "map.png"))
    if full_map is None:
        print("✗ Failed to load map!")
        return

    print(f"\nLoaded data:")
    print(f"  Train rough predictions: {len(train_rough_dict)}")
    print(f"  Test rough predictions:  {len(test_rough_dict)}")
    print(f"  Train ground truth:      {len(gt_dict)}")
    print(f"  Map size:                {full_map.shape[1]}x{full_map.shape[0]}")

    # Create a mapping of image IDs to their type (train or test)
    image_types = {}
    for img_id in train_rough_dict.keys():
        image_types[img_id] = 'train'
    for img_id in test_rough_dict.keys():
        image_types[img_id] = 'test'

    print(f"\nTotal images to process: {len(image_types)} ({len(train_rough_dict)} train + {len(test_rough_dict)} test)")

    # Process all images
    train_successful = 0
    train_failed = 0
    test_successful = 0
    test_failed = 0

    all_image_ids = sorted(image_types.keys())

    # Process all images with progress bar
    pbar = tqdm(all_image_ids, desc="Processing images", unit="img")
    for img_id in pbar:
        img_type = image_types[img_id]
        rough_dict = train_rough_dict if img_type == 'train' else test_rough_dict

        # Update progress bar description
        pbar.set_description(f"Processing {img_id:04d} ({img_type})")

        try:
            success = collect_matches_for_image(
                img_id, root, rough_dict, gt_dict, full_map, image_type=img_type
            )
            if success:
                if img_type == 'train':
                    train_successful += 1
                else:
                    test_successful += 1
            else:
                if img_type == 'train':
                    train_failed += 1
                else:
                    test_failed += 1
        except Exception as e:
            print(f"\n✗ Error processing image {img_id:04d}: {e}")
            import traceback
            traceback.print_exc()
            if img_type == 'train':
                train_failed += 1
            else:
                test_failed += 1

        # Update progress bar postfix with current stats
        pbar.set_postfix({
            'train_ok': train_successful,
            'train_fail': train_failed,
            'test_ok': test_successful,
            'test_fail': test_failed
        })

    # Final summary
    total_train = len(train_rough_dict)
    total_test = len(test_rough_dict)
    total_all = len(image_types)

    print("\n" + "="*80)
    print("BATCH COLLECTION COMPLETE")
    print("="*80)
    print(f"\nTrain Images:")
    print(f"  Successful: {train_successful}/{total_train}")
    print(f"  Failed:     {train_failed}")
    print(f"\nTest Images:")
    print(f"  Successful: {test_successful}/{total_test}")
    print(f"  Failed:     {test_failed}")
    print(f"\nTotal:")
    print(f"  Successful: {train_successful + test_successful}/{total_all}")
    print(f"  Failed:     {train_failed + test_failed}")
    print("\nNext steps:")
    print("  1. Review visualizations in match_visualizations/img_XXXX/")
    print("  2. Run analyze_matches.py on each image to tune position calculation")
    print("  3. Compare methods across all images to find the most robust approach")
    print("="*80)


if __name__ == "__main__":
    main()
