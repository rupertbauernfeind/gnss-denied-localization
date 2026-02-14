#!/usr/bin/env python3
"""
Analyze winter condition impact on GNSS-denied localization.

Compares localization performance between original training images
and winter images taken from the same positions.
"""

import sys
import csv
import math
import os
from pathlib import Path
import tempfile

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # skip torch.compile

import cv2
import numpy as np
from PIL import Image
import torch

# Import MASt3R (installed via vismatch package)
sys.path.insert(0, str(Path.home() / ".local/lib/python3.10/site-packages/vismatch/third_party/mast3r"))
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs

# Import similarity refinement method
from analyze_matches import method_similarity_ransac_thresh

# Configuration
WINTER_IDS = [13, 15, 40]
CROP_SIZE = 750
MATCH_SIZE = 512
MAP_W, MAP_H = 5000, 2500
MAX_MOVEMENT = 140.0
MIN_INLIERS = 20
SIMILARITY_THRESHOLD = 1.0


def load_ground_truth(path):
    """Load ground truth positions from train_pos.csv."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def load_training_errors(path):
    """Load original training errors from train_analysis.csv."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            img_id = int(row["id"])
            error_px = float(row["error_px"])
            data[img_id] = error_px
    return data


def load_rough_positions(path):
    """Load rough GPS positions from rough_matching/train_predictions.csv."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            img_id = int(row["id"])
            if img_id in WINTER_IDS:
                data[img_id] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def initialize_mast3r():
    """Initialize MASt3R model on GPU or CPU."""
    print("Initializing MASt3R model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    print("  Model loaded successfully")
    return model, device


def process_winter_image(img_id, rough_x, rough_y, gt_x, gt_y, full_map, model, device):
    """
    Run complete MASt3R matching pipeline for one winter image.
    Adapted from refine.py:98-236

    IMPORTANT: Uses rough GPS position (not ground truth) to extract map crop,
    matching the exact pipeline used for training images in refine.py

    Returns: match_data dict or None if matching failed
    """
    print(f"\nProcessing winter image {img_id:04d}...")

    # 1. Load winter image
    script_dir = Path(__file__).parent
    winter_img_path = script_dir / "winter_images" / f"{img_id:04d}_winter.png"

    print(f"  Loading {winter_img_path}")
    drone = cv2.imread(str(winter_img_path))
    if drone is None:
        print(f"  ✗ Failed to load winter image")
        return None

    # 2. Extract map crop around ROUGH GPS position (not ground truth!)
    # This matches the pipeline in refine.py which uses rough GPS
    cx = int(round(rough_x))
    cy = int(round(rough_y))
    x0 = max(0, min(MAP_W - CROP_SIZE, cx - CROP_SIZE // 2))
    y0 = max(0, min(MAP_H - CROP_SIZE, cy - CROP_SIZE // 2))
    map_crop = full_map[y0 : y0 + CROP_SIZE, x0 : x0 + CROP_SIZE]
    crop_offset = (x0, y0)

    print(f"  Rough GPS position: ({rough_x:.2f}, {rough_y:.2f})")
    print(f"  Ground truth position: ({gt_x:.2f}, {gt_y:.2f})")
    print(f"  Map crop offset: ({x0}, {y0}) [from rough GPS]")

    # 3. Pre-resize drone to 512px width
    h_orig, w_orig = drone.shape[:2]
    mast3r_scale = MATCH_SIZE / w_orig
    drone_resized = cv2.resize(drone, (MATCH_SIZE, int(h_orig * mast3r_scale)))

    # Save pre-resized dimensions for center calculation
    h_pre, w_pre = drone_resized.shape[:2]

    # 4. BGR -> RGB for MASt3R
    drone_rgb = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(map_crop, cv2.COLOR_BGR2RGB)

    # 5. Convert to PIL and save to temp files
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
        print(f"  Running MASt3R matching...")
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

        # Filter by border (3px)
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
                            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
                            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches]
        matches_im1 = matches_im1[valid_matches]

        print(f"  Matches found: {len(matches_im0)}")

        if len(matches_im0) < 4:
            print(f"  ✗ Insufficient matches (<4)")
            return None

        # 9. Compute homography with RANSAC
        src_pts = matches_im0.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = matches_im1.reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            print(f"  ✗ Homography estimation failed")
            return None

        n_inliers = int(mask.sum())
        print(f"  RANSAC inliers (homography): {n_inliers}")

        if n_inliers < MIN_INLIERS:
            print(f"  ✗ Insufficient inliers (<{MIN_INLIERS})")
            return None

        # 10. Store match data
        match_data = {
            'kp_drone': matches_im0,
            'kp_crop': matches_im1,
            'H': H,
            'mask': mask,
            'crop_offset': crop_offset,
            'drone_shape': (h_pre, w_pre),  # Use pre-resized dimensions
            'rough_position': (rough_x, rough_y),
            'gt_position': (gt_x, gt_y),
            'n_inliers': n_inliers
        }

        return match_data

    finally:
        # Clean up temporary files
        if os.path.exists(drone_path):
            os.unlink(drone_path)
        if os.path.exists(crop_path):
            os.unlink(crop_path)


def apply_similarity_refinement(match_data):
    """
    Apply Similarity thresh=1.0 refinement method.
    Calls method_similarity_ransac_thresh from analyze_matches.py

    IMPORTANT: Validates movement from ROUGH GPS (not ground truth),
    matching the exact validation used in batch_analyze.py lines 421-441

    Returns: (final_x, final_y, n_inliers_sim, success)
    """
    kp_drone = match_data['kp_drone']
    kp_crop = match_data['kp_crop']
    H = match_data['H']
    mask = match_data['mask']
    crop_offset = match_data['crop_offset']
    drone_shape = match_data['drone_shape']
    rough_x, rough_y = match_data['rough_position']
    gt_x, gt_y = match_data['gt_position']

    # Call similarity refinement method
    x_refined, y_refined, description = method_similarity_ransac_thresh(
        kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=SIMILARITY_THRESHOLD
    )

    if x_refined is None:
        print(f"  ✗ Similarity refinement failed - using rough GPS as fallback")
        # Use rough GPS as fallback (matches batch_analyze.py behavior)
        return rough_x, rough_y, 0, False

    # Extract number of inliers from description
    # Format: "Similarity thresh=1.0 (XXX inliers)"
    import re
    match = re.search(r'\((\d+) inliers\)', description)
    n_inliers_sim = int(match.group(1)) if match else 0

    print(f"  Similarity inliers: {n_inliers_sim}")
    print(f"  Refined position: ({x_refined:.1f}, {y_refined:.1f})")

    # Validate movement from ROUGH GPS (not ground truth!)
    # This matches the validation in batch_analyze.py lines 421-441
    movement_from_rough = math.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)
    print(f"  Movement from rough GPS: {movement_from_rough:.1f}px")

    if movement_from_rough > MAX_MOVEMENT:
        print(f"  ⚠ Warning: Moved {movement_from_rough:.1f}px from rough GPS (>{MAX_MOVEMENT}px threshold)")
        print(f"  Using rough GPS as fallback")
        # Use rough GPS as fallback (matches batch_analyze.py behavior)
        error_rough = calculate_error(rough_x, rough_y, gt_x, gt_y)
        print(f"  Error from GT (using rough GPS): {error_rough:.1f}px")
        return rough_x, rough_y, n_inliers_sim, False

    # Refinement validated - use refined position
    error_refined = calculate_error(x_refined, y_refined, gt_x, gt_y)
    print(f"  Error from GT (using refinement): {error_refined:.1f}px")
    print(f"  ✓ Refinement passed validation")

    return x_refined, y_refined, n_inliers_sim, True


def calculate_error(x, y, gt_x, gt_y):
    """Calculate Euclidean error in pixels."""
    return math.sqrt((x - gt_x)**2 + (y - gt_y)**2)


def save_results_to_csv(results, output_path):
    """Save comparison results to CSV."""
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'id', 'original_error_px', 'winter_error_px', 'degradation_px',
            'degradation_percent', 'n_matches', 'n_inliers_homography',
            'n_inliers_similarity', 'success', 'method_description'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved to: {output_path}")


def print_comparison_summary(results):
    """Print detailed comparison summary."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    successful_results = [r for r in results if r['success']]
    results_with_data = [r for r in results if r['winter_error_px'] != float('inf')]

    for r in results:
        print(f"\nImage {r['id']}:")
        print(f"  Original error:    {r['original_error_px']:.1f} px")

        if r['winter_error_px'] != float('inf'):
            print(f"  Winter error:      {r['winter_error_px']:.1f} px")
            print(f"  Degradation:       {r['degradation_px']:+.1f} px ({r['degradation_percent']:+.1f}%)")
            print(f"  Matches:           {r['n_matches']} ({r['n_inliers_homography']} H inliers, {r['n_inliers_similarity']} sim inliers)")
            if r['success']:
                print(f"  Status:            SUCCESS (passed validation)")
            else:
                print(f"  Status:            FAILED VALIDATION (but method produced result)")
        else:
            print(f"  Winter error:      FAILED (no result)")
            print(f"  Status:            COMPLETE FAILURE")

    if results_with_data:
        avg_original = sum(r['original_error_px'] for r in results_with_data) / len(results_with_data)
        avg_winter = sum(r['winter_error_px'] for r in results_with_data) / len(results_with_data)
        avg_degradation = sum(r['degradation_px'] for r in results_with_data) / len(results_with_data)
        avg_degradation_pct = sum(r['degradation_percent'] for r in results_with_data) / len(results_with_data)

        print(f"\nAGGREGATE STATISTICS (all results with data):")
        print(f"  Average original error:   {avg_original:.2f} px")
        print(f"  Average winter error:     {avg_winter:.2f} px")
        print(f"  Average degradation:      {avg_degradation:+.2f} px ({avg_degradation_pct:+.1f}%)")
        print(f"  Validation pass rate:     {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.0f}%)")
        print(f"  Method completion rate:   {len(results_with_data)}/{len(results)} ({len(results_with_data)/len(results)*100:.0f}%)")

        # Assess impact
        if avg_degradation_pct < 20:
            impact = "MINIMAL"
        elif avg_degradation_pct < 50:
            impact = "MILD"
        elif avg_degradation_pct < 100:
            impact = "MODERATE"
        elif avg_degradation_pct < 200:
            impact = "SIGNIFICANT"
        else:
            impact = "SEVERE"

        # Assess robustness based on both accuracy AND success rate
        if len(successful_results) == len(results) and avg_degradation_pct < 40:
            robustness = "EXCELLENT"
        elif len(results_with_data) == len(results) and avg_degradation_pct < 100:
            robustness = "GOOD"
        elif len(results_with_data) >= len(results) * 0.5:
            robustness = "FAIR"
        else:
            robustness = "POOR"

        print(f"\nCONCLUSIONS:")
        print(f"  - Winter conditions impact: {impact}")
        print(f"  - Method robustness:        {robustness}")

        if robustness == "EXCELLENT":
            print(f"  - Recommendation:           Method performs well on winter images")
        elif robustness == "GOOD":
            print(f"  - Recommendation:           Method works but produces larger errors or fails validation")
        elif robustness == "FAIR":
            print(f"  - Recommendation:           Winter significantly degrades performance, consider fallback strategies")
        else:
            print(f"  - Recommendation:           Method struggles with winter conditions, use alternative approach")
    else:
        print(f"\nAll images failed completely - winter conditions too severe for matching")

    print("="*80)


def main():
    print("="*80)
    print("WINTER CONDITION ANALYSIS")
    print("="*80)

    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    gt_path = repo_root / "data" / "train_data" / "train_pos.csv"
    rough_path = repo_root / "rough_matching" / "train_predictions.csv"
    training_errors_path = script_dir / "train_analysis.csv"
    map_path = repo_root / "data" / "map.png"
    output_csv = script_dir / "winter_analysis_results.csv"

    # Load data
    print("\nLoading data...")
    gt_dict = load_ground_truth(gt_path)
    rough_dict = load_rough_positions(rough_path)
    training_errors = load_training_errors(training_errors_path)
    full_map = cv2.imread(str(map_path))

    if full_map is None:
        print(f"✗ Failed to load map from {map_path}")
        return

    print(f"  Ground truth positions: {len(gt_dict)} images")
    print(f"  Rough GPS positions: {len(rough_dict)} images")
    print(f"  Training errors: {len(training_errors)} images")
    print(f"  Map size: {full_map.shape}")

    # Initialize MASt3R
    model, device = initialize_mast3r()

    # Process each winter image
    results = []
    for img_id in WINTER_IDS:
        gt_x, gt_y = gt_dict[img_id]
        rough_x, rough_y = rough_dict[img_id]
        original_error = training_errors[img_id]

        # Show rough GPS offset
        rough_offset = math.sqrt((rough_x - gt_x)**2 + (rough_y - gt_y)**2)
        print(f"\nImage {img_id}: Rough GPS offset from GT = {rough_offset:.1f}px")

        # Run MASt3R matching (using rough GPS for crop extraction, like in training)
        match_data = process_winter_image(img_id, rough_x, rough_y, gt_x, gt_y, full_map, model, device)

        if match_data is None:
            # Matching failed
            results.append({
                'id': img_id,
                'original_error_px': original_error,
                'winter_error_px': float('inf'),
                'degradation_px': float('inf'),
                'degradation_percent': float('inf'),
                'n_matches': 0,
                'n_inliers_homography': 0,
                'n_inliers_similarity': 0,
                'success': False,
                'method_description': 'MASt3R matching failed'
            })
            continue

        # Apply refinement
        refined_x, refined_y, n_inliers_sim, validation_passed = apply_similarity_refinement(match_data)

        if refined_x is None:
            # Refinement completely failed
            results.append({
                'id': img_id,
                'original_error_px': original_error,
                'winter_error_px': float('inf'),
                'degradation_px': float('inf'),
                'degradation_percent': float('inf'),
                'n_matches': len(match_data['kp_drone']),
                'n_inliers_homography': match_data['n_inliers'],
                'n_inliers_similarity': n_inliers_sim,
                'success': False,
                'method_description': 'Similarity refinement failed'
            })
            continue

        # Calculate error (even if validation failed, to show actual performance)
        winter_error = calculate_error(refined_x, refined_y, gt_x, gt_y)
        degradation = winter_error - original_error
        degradation_pct = (degradation / original_error) * 100 if original_error > 0 else 0

        print(f"  Winter error: {winter_error:.1f} px")
        print(f"  Original error: {original_error:.1f} px")
        print(f"  Degradation: {degradation:+.1f} px ({degradation_pct:+.1f}%)")

        # Determine status message
        if validation_passed:
            status_msg = f'Similarity thresh={SIMILARITY_THRESHOLD} ({n_inliers_sim} inliers)'
        else:
            status_msg = f'Similarity thresh={SIMILARITY_THRESHOLD} ({n_inliers_sim} inliers) - FAILED VALIDATION'

        results.append({
            'id': img_id,
            'original_error_px': original_error,
            'winter_error_px': winter_error,
            'degradation_px': degradation,
            'degradation_percent': degradation_pct,
            'n_matches': len(match_data['kp_drone']),
            'n_inliers_homography': match_data['n_inliers'],
            'n_inliers_similarity': n_inliers_sim,
            'success': validation_passed,  # Success only if validation passed
            'method_description': status_msg
        })

    # Generate outputs
    save_results_to_csv(results, output_csv)
    print_comparison_summary(results)


if __name__ == "__main__":
    main()
