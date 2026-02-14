#!/usr/bin/env python3
"""
Batch analyze all collected matches across multiple images.

This script runs all position calculation methods on all collected images
and reports aggregate statistics to find the best method.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
import random
from tqdm import tqdm

# Import all position calculation methods from analyze_matches
from analyze_matches import (
    method_homography_center,
    method_median_transform,
    method_mean_transform,
    method_weighted_by_distance,
    method_homography_corners,
    method_direct_correspondence,
    method_top_k_nearest,
    method_median_k_nearest,
    method_quadratic_weighted,
    method_exponential_weighted,
    method_radius_filter,
    method_median_displacement,
    method_mean_displacement,
    method_weighted_displacement,
    method_affine_ransac,
    method_similarity_ransac,
    method_affine_median,
    method_homography_strict_ransac,
    method_local_homography,
    method_weighted_homography,
    method_pnp_solver,
    method_pnp_hybrid,
    method_pycolmap_ba,
    method_reproj_filtered_homography,
    method_normalized_homography,
    method_irls_homography,
    method_local_weighted_homography,
    method_symmetric_transfer_homography,
    method_similarity_ransac_thresh,
    method_reproj_filtered_similarity,
    method_iterative_similarity,
    method_weighted_similarity,
    method_local_similarity,
)


def analyze_one_image_matcher(match_file, method, method_name, rough_error):
    """
    Analyze one image+matcher combination with one position calculation method.

    Returns: (error_px, success) where success is True if method succeeded
    If method fails OR moves too far from rough GPS, returns rough GPS error as fallback.
    This matches the validation logic used in generate_refined_predictions().
    """
    MAX_MOVEMENT = 140.0  # Same threshold as in generate_refined_predictions()

    if not match_file.exists():
        return rough_error, False

    # Load match data
    data = np.load(match_file)
    kp_drone = data['kp_drone']
    kp_crop = data['kp_crop']
    H = data['H']
    mask = data['mask']
    crop_offset = tuple(data['crop_offset'])
    drone_shape = tuple(data['drone_shape'])
    gt_x, gt_y = data['gt_position']
    rough_x, rough_y = data['rough_position']

    # Run the position calculation method
    x_refined, y_refined, description = method(kp_drone, kp_crop, H, mask, crop_offset, drone_shape)

    if x_refined is None:
        # Method failed - use rough GPS error as fallback for fair comparison
        return rough_error, False

    # Validation: reject refinements that move more than MAX_MOVEMENT from rough GPS
    movement = np.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)

    if movement > MAX_MOVEMENT:
        # Refinement moved too far - likely an error, use rough GPS as fallback
        return rough_error, False

    # Refinement looks good - calculate error against ground truth
    error = np.sqrt((x_refined - gt_x)**2 + (y_refined - gt_y)**2)
    return error, True


def main():
    """Batch analyze all images with all methods."""

    # Configuration
    # Process ALL train images (IDs 13-2831 that have both match data and ground truth)
    root_dir = Path(__file__).parent
    match_dir_root = root_dir / "match_visualizations"

    # Find all image IDs that have match data with ground truth
    all_train_ids = []
    for img_dir in sorted(match_dir_root.glob("img_*")):
        img_id = int(img_dir.name.split("_")[1])
        match_file = img_dir / "master_matches.npz"
        if match_file.exists():
            # Check if it has ground truth (train image)
            data = np.load(match_file)
            if 'gt_position' in data:
                all_train_ids.append(img_id)

    print(f"Found {len(all_train_ids)} train images with match data and ground truth")
    print(f"ID range: {min(all_train_ids)} to {max(all_train_ids)}")

    # Random sample of 100 images (or all if less than 100)
    sample_size = min(100, len(all_train_ids))
    random.seed(42)  # For reproducibility
    image_ids = sorted(random.sample(all_train_ids, sample_size))

    print(f"Using random sample of {sample_size} images for analysis")

    matcher = 'master'  # Only Master matcher

    # Define methods to test (using lambdas for parameterized methods)
    methods_to_test = [
        ("Homography center", method_homography_center),
        ("Median transform", method_median_transform),
        ("Mean transform", method_mean_transform),
        ("Weighted by distance", method_weighted_by_distance),
        ("Homography corners", method_homography_corners),
        ("Direct correspondence", method_direct_correspondence),
        ("Top-3 nearest avg", lambda *args: method_top_k_nearest(*args, k=3)),
        ("Top-5 nearest avg", lambda *args: method_top_k_nearest(*args, k=5)),
        ("Top-10 nearest avg", lambda *args: method_top_k_nearest(*args, k=10)),
        ("Median top-5", lambda *args: method_median_k_nearest(*args, k=5)),
        ("Median top-10", lambda *args: method_median_k_nearest(*args, k=10)),
        ("Median top-20", lambda *args: method_median_k_nearest(*args, k=20)),
        ("Quadratic weighted", method_quadratic_weighted),
        ("Exponential weighted", method_exponential_weighted),
        ("Radius filter r=50", lambda *args: method_radius_filter(*args, radius=50)),
        ("Radius filter r=100", lambda *args: method_radius_filter(*args, radius=100)),
        ("Radius filter r=150", lambda *args: method_radius_filter(*args, radius=150)),
        ("Median displacement", method_median_displacement),
        ("Mean displacement", method_mean_displacement),
        ("Weighted displacement", method_weighted_displacement),
        ("Affine RANSAC", method_affine_ransac),
        ("Similarity RANSAC", method_similarity_ransac),
        ("Affine median", method_affine_median),
        # Similarity variations (based on best performer)
        ("Similarity thresh=1.0", lambda *args: method_similarity_ransac_thresh(*args, threshold=1.0)),
        ("Similarity thresh=1.5", lambda *args: method_similarity_ransac_thresh(*args, threshold=1.5)),
        ("Similarity thresh=3.0", lambda *args: method_similarity_ransac_thresh(*args, threshold=3.0)),
        ("Reproj Sim thresh=1.0", lambda *args: method_reproj_filtered_similarity(*args, threshold=1.0)),
        ("Reproj Sim thresh=1.5", lambda *args: method_reproj_filtered_similarity(*args, threshold=1.5)),
        ("Reproj Sim thresh=2.0", lambda *args: method_reproj_filtered_similarity(*args, threshold=2.0)),
        ("Iterative Sim n=2", lambda *args: method_iterative_similarity(*args, n_iterations=2)),
        ("Iterative Sim n=3", lambda *args: method_iterative_similarity(*args, n_iterations=3)),
        ("Weighted Sim σ=75", lambda *args: method_weighted_similarity(*args, sigma=75)),
        ("Weighted Sim σ=100", lambda *args: method_weighted_similarity(*args, sigma=100)),
        ("Local Sim r=75", lambda *args: method_local_similarity(*args, radius=75)),
        ("Local Sim r=100", lambda *args: method_local_similarity(*args, radius=100)),
        ("Local Sim r=125", lambda *args: method_local_similarity(*args, radius=125)),
        ("Homography strict RANSAC", method_homography_strict_ransac),
        ("Local H r=75", lambda *args: method_local_homography(*args, radius=75)),
        ("Local H r=100", lambda *args: method_local_homography(*args, radius=100)),
        ("Local H r=125", lambda *args: method_local_homography(*args, radius=125)),
        ("Weighted H σ=75", lambda *args: method_weighted_homography(*args, sigma=75)),
        ("Weighted H σ=100", lambda *args: method_weighted_homography(*args, sigma=100)),
        ("Weighted H σ=125", lambda *args: method_weighted_homography(*args, sigma=125)),
        ("PnP RANSAC", lambda *args: method_pnp_solver(*args, method_name="RANSAC")),
        ("PnP ITERATIVE", lambda *args: method_pnp_solver(*args, method_name="ITERATIVE")),
        ("PnP DLS", lambda *args: method_pnp_solver(*args, method_name="DLS")),
        ("PnP hybrid", method_pnp_hybrid),
        ("PyCOLMAP BA", method_pycolmap_ba),
        # New robust variations (filtering & relaxed assumptions)
        ("Reproj filter thresh=1.0", lambda *args: method_reproj_filtered_homography(*args, threshold=1.0)),
        ("Reproj filter thresh=2.0", lambda *args: method_reproj_filtered_homography(*args, threshold=2.0)),
        ("Reproj filter thresh=3.0", lambda *args: method_reproj_filtered_homography(*args, threshold=3.0)),
        ("Normalized H", method_normalized_homography),
        ("IRLS H (3 iter)", lambda *args: method_irls_homography(*args, n_iterations=3)),
        ("IRLS H (5 iter)", lambda *args: method_irls_homography(*args, n_iterations=5)),
        ("Local+Weighted r=100,σ=50", lambda *args: method_local_weighted_homography(*args, radius=100, sigma=50)),
        ("Local+Weighted r=100,σ=75", lambda *args: method_local_weighted_homography(*args, radius=100, sigma=75)),
        ("Symmetric H", method_symmetric_transfer_homography),
    ]

    # Collect all results: method_name -> [errors]
    results = defaultdict(list)
    success_counts = defaultdict(int)  # Track how many succeeded per method
    rough_errors = []

    # Process each image with progress bar
    print("\nAnalyzing images...")
    for img_id in tqdm(image_ids, desc="Processing images", unit="img"):
        match_dir = match_dir_root / f"img_{img_id:04d}"
        match_file = match_dir / f"{matcher}_matches.npz"

        if not match_file.exists():
            continue

        # Load ground truth and rough GPS for this image
        data = np.load(match_file)
        gt_x, gt_y = data['gt_position']
        rough_x, rough_y = data['rough_position']
        rough_err = np.sqrt((rough_x - gt_x)**2 + (rough_y - gt_y)**2)
        rough_errors.append(rough_err)

        # Test each method
        for method_name, method_func in methods_to_test:
            error, success = analyze_one_image_matcher(match_file, method_func, method_name, rough_err)

            # Always record error (uses rough GPS as fallback if method failed)
            results[method_name].append(error)
            if success:
                success_counts[method_name] += 1

    # Calculate statistics
    print("\n" + "="*100)
    print("BATCH ANALYSIS RESULTS")
    print(f"Images analyzed: {len(rough_errors)} (random sample from {len(all_train_ids)} total train images)")
    print(f"Matcher: {matcher}")
    print(f"Methods tested: {len(methods_to_test)}")
    print("="*100)

    # Rough GPS baseline
    if rough_errors:
        print(f"\nROUGH GPS BASELINE:")
        print(f"  Mean error: {np.mean(rough_errors):.1f}px (±{np.std(rough_errors):.1f}px)")
        print(f"  Median error: {np.median(rough_errors):.1f}px")
        print(f"  Min/Max: {np.min(rough_errors):.1f}px / {np.max(rough_errors):.1f}px")

    # Analyze each method
    method_stats = []

    for method_name, method_func in methods_to_test:
        errors = results[method_name]

        if len(errors) == 0:
            continue

        mean_err = np.mean(errors)
        median_err = np.median(errors)
        std_err = np.std(errors)
        min_err = np.min(errors)
        max_err = np.max(errors)
        n_success = len(errors)

        # Competition score: mean of accuracies at 5m, 25m, 100m
        acc_5m = sum(1 for e in errors if e <= 25) / len(errors) * 100
        acc_25m = sum(1 for e in errors if e <= 125) / len(errors) * 100
        acc_100m = sum(1 for e in errors if e <= 500) / len(errors) * 100
        comp_score = (acc_5m + acc_25m + acc_100m) / 3

        # Get success count (predictions within 140px of rough GPS)
        n_accepted = success_counts[method_name]
        acceptance_rate = n_accepted / n_success * 100 if n_success > 0 else 0

        method_stats.append({
            'name': method_name,
            'mean': mean_err,
            'median': median_err,
            'std': std_err,
            'min': min_err,
            'max': max_err,
            'n': n_success,
            'n_accepted': n_accepted,
            'acceptance_rate': acceptance_rate,
            'score': comp_score,
            'acc_5m': acc_5m,
            'acc_25m': acc_25m,
            'acc_100m': acc_100m,
        })

    # Sort by COMPETITION SCORE (descending - higher is better)
    method_stats.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "="*135)
    print("METHODS RANKED BY COMPETITION SCORE (Best to Worst)")
    print("="*135)
    print(f"{'Rank':<6} {'Method':<30} {'Median':<10} {'Mean':<10} {'Score':<10} {'@5m':<8} {'@25m':<8} {'@100m':<8} {'Accept%':<9} {'N':<6}")
    print("-"*135)

    for rank, stats in enumerate(method_stats, 1):
        print(f"{rank:<6} {stats['name']:<30} "
              f"{stats['median']:>8.1f}px "
              f"{stats['mean']:>8.1f}px "
              f"{stats['score']:>8.2f}  "
              f"{stats['acc_5m']:>6.1f}% "
              f"{stats['acc_25m']:>6.1f}% "
              f"{stats['acc_100m']:>6.1f}% "
              f"{stats['acceptance_rate']:>7.1f}% "
              f"{stats['n']:>5}")

    # Summary
    print("\n" + "="*135)
    print("SUMMARY")
    print("="*135)

    if method_stats:
        best_by_score = method_stats[0]  # First in list (sorted by score descending)
        best_by_median = min(method_stats, key=lambda x: x['median'])

        baseline_median = np.median(rough_errors) if rough_errors else 0
        baseline_mean = np.mean(rough_errors) if rough_errors else 0

        # Baseline competition score
        baseline_acc_5m = sum(1 for e in rough_errors if e <= 25) / len(rough_errors) * 100
        baseline_acc_25m = sum(1 for e in rough_errors if e <= 125) / len(rough_errors) * 100
        baseline_acc_100m = sum(1 for e in rough_errors if e <= 500) / len(rough_errors) * 100
        baseline_score = (baseline_acc_5m + baseline_acc_25m + baseline_acc_100m) / 3

        print(f"\nBEST BY COMPETITION SCORE: {best_by_score['name']}")
        print(f"  Competition score: {best_by_score['score']:.2f} (vs {baseline_score:.2f} rough GPS)")
        print(f"    Accuracy @5m:   {best_by_score['acc_5m']:.1f}% (vs {baseline_acc_5m:.1f}%)")
        print(f"    Accuracy @25m:  {best_by_score['acc_25m']:.1f}% (vs {baseline_acc_25m:.1f}%)")
        print(f"    Accuracy @100m: {best_by_score['acc_100m']:.1f}% (vs {baseline_acc_100m:.1f}%)")
        print(f"  Median error: {best_by_score['median']:.1f}px (vs {baseline_median:.1f}px rough GPS)")
        print(f"  Mean error:   {best_by_score['mean']:.1f}px (vs {baseline_mean:.1f}px rough GPS)")
        print(f"  Acceptance rate: {best_by_score['acceptance_rate']:.1f}% ({best_by_score['n_accepted']}/{best_by_score['n']} within 140px of rough GPS)")

        if best_by_median['name'] != best_by_score['name']:
            print(f"\nBEST BY MEDIAN ERROR: {best_by_median['name']}")
            print(f"  Median error: {best_by_median['median']:.1f}px (vs {baseline_median:.1f}px rough GPS)")
            print(f"  Competition score: {best_by_median['score']:.2f} (vs {baseline_score:.2f} rough GPS)")
            print(f"  Acceptance rate: {best_by_median['acceptance_rate']:.1f}% ({best_by_median['n_accepted']}/{best_by_median['n']})")

        median_improvement = baseline_median - best_by_median['median']
        score_improvement = best_by_score['score'] - baseline_score

        print(f"\n📊 VALIDATION THRESHOLD: 140px (predictions moving >140px from rough GPS use rough GPS instead)")

        if best_by_score['median'] < 10:
            print(f"\n✅ GOAL ACHIEVED! Median error {best_by_score['median']:.1f}px < 10px target!")
        elif score_improvement > 0:
            print(f"\n✓ Improvement: +{score_improvement:.2f} score points ({score_improvement/baseline_score*100:.1f}%), {baseline_median - best_by_score['median']:.1f}px median reduction")
        else:
            print(f"\n✗ Not yet improving over baseline")

    print("="*135)


def generate_refined_predictions():
    """
    Generate refined predictions for all train and test images using Homography center method.

    Reads match data from match_visualizations/img_XXXX/master_matches.npz
    and writes refined predictions to:
      - refine_matching/train_predictions.csv
      - refine_matching/test_predicted.csv
    """
    import csv

    print("="*100)
    print("GENERATING REFINED PREDICTIONS")
    print("Method: Homography center")
    print("="*100)

    root_dir = Path(__file__).parent.parent
    match_dir_root = Path(__file__).parent / "match_visualizations"

    # Load rough predictions to know which images exist
    train_rough_path = root_dir / "rough_matching" / "train_predictions.csv"
    test_rough_path = root_dir / "rough_matching" / "test_predicted.csv"

    def load_image_ids(csv_path):
        """Load image IDs from CSV."""
        ids = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(int(row['id']))
        return sorted(ids)

    train_ids = load_image_ids(train_rough_path)
    test_ids = load_image_ids(test_rough_path)

    print(f"\nFound {len(train_ids)} train images")
    print(f"Found {len(test_ids)} test images")
    print(f"Total: {len(train_ids) + len(test_ids)} images\n")

    # Process train images
    train_results = []
    train_successful = 0
    train_failed = 0

    print("Processing train images...")
    for img_id in train_ids:
        match_file = match_dir_root / f"img_{img_id:04d}" / "master_matches.npz"

        if not match_file.exists():
            print(f"  ✗ Image {img_id:04d}: No match data found")
            train_failed += 1
            continue

        # Load match data
        data = np.load(match_file)
        kp_drone = data['kp_drone']
        kp_crop = data['kp_crop']
        H = data['H']
        mask = data['mask']
        crop_offset = tuple(data['crop_offset'])
        drone_shape = tuple(data['drone_shape'])
        rough_x, rough_y = data['rough_position']

        # Apply homography center method
        x_refined, y_refined, description = method_homography_center(
            kp_drone, kp_crop, H, mask, crop_offset, drone_shape
        )

        # Validation: reject refinements that move more than 140px from rough GPS
        MAX_MOVEMENT = 140.0

        if x_refined is None:
            # Method failed - use rough GPS as fallback
            print(f"  ✗ Image {img_id:04d}: Method failed, using rough GPS")
            x_map, y_map = rough_x, rough_y
            train_failed += 1
        else:
            # Check if refinement moved too far from rough GPS
            movement = np.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)

            if movement > MAX_MOVEMENT:
                # Refinement moved too far - likely an error, use rough GPS
                print(f"  ⚠ Image {img_id:04d}: Moved {movement:.1f}px > {MAX_MOVEMENT}px, using rough GPS")
                x_map, y_map = rough_x, rough_y
                train_failed += 1
            else:
                # Refinement looks good - use it
                x_map, y_map = x_refined, y_refined
                train_successful += 1

                # Calculate improvement over rough (if ground truth available)
                if 'gt_position' in data:
                    gt_x, gt_y = data['gt_position']
                    rough_err = np.sqrt((rough_x - gt_x)**2 + (rough_y - gt_y)**2)
                    refined_err = np.sqrt((x_map - gt_x)**2 + (y_map - gt_y)**2)
                    improvement = rough_err - refined_err
                    print(f"  ✓ Image {img_id:04d}: {rough_err:.1f}px → {refined_err:.1f}px (Δ={improvement:+.1f}px, moved {movement:.1f}px)")

        train_results.append({'id': img_id, 'x_pixel': x_map, 'y_pixel': y_map})

    # Process test images
    test_results = []
    test_successful = 0
    test_failed = 0

    print("\nProcessing test images...")
    for img_id in test_ids:
        match_file = match_dir_root / f"img_{img_id:04d}" / "master_matches.npz"

        if not match_file.exists():
            print(f"  ✗ Image {img_id:04d}: No match data found")
            test_failed += 1
            continue

        # Load match data
        data = np.load(match_file)
        kp_drone = data['kp_drone']
        kp_crop = data['kp_crop']
        H = data['H']
        mask = data['mask']
        crop_offset = tuple(data['crop_offset'])
        drone_shape = tuple(data['drone_shape'])
        rough_x, rough_y = data['rough_position']

        # Apply homography center method
        x_refined, y_refined, description = method_homography_center(
            kp_drone, kp_crop, H, mask, crop_offset, drone_shape
        )

        # Validation: reject refinements that move more than 140px from rough GPS
        MAX_MOVEMENT = 140.0

        if x_refined is None:
            # Method failed - use rough GPS as fallback
            print(f"  ✗ Image {img_id:04d}: Method failed, using rough GPS")
            x_map, y_map = rough_x, rough_y
            test_failed += 1
        else:
            # Check if refinement moved too far from rough GPS
            movement = np.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)

            if movement > MAX_MOVEMENT:
                # Refinement moved too far - likely an error, use rough GPS
                print(f"  ⚠ Image {img_id:04d}: Moved {movement:.1f}px > {MAX_MOVEMENT}px, using rough GPS")
                x_map, y_map = rough_x, rough_y
                test_failed += 1
            else:
                # Refinement looks good - use it
                x_map, y_map = x_refined, y_refined
                test_successful += 1
                print(f"  ✓ Image {img_id:04d}: Refined position ({x_map:.1f}, {y_map:.1f}), moved {movement:.1f}px")

        test_results.append({'id': img_id, 'x_pixel': x_map, 'y_pixel': y_map})

    # Write train predictions
    train_output = Path(__file__).parent / "train_predictions.csv"
    with open(train_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        writer.writerows(train_results)

    print(f"\n✓ Wrote train predictions to: {train_output}")

    # Write test predictions
    test_output = Path(__file__).parent / "test_predicted.csv"
    with open(test_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        writer.writerows(test_results)

    print(f"✓ Wrote test predictions to: {test_output}")

    # Summary
    print("\n" + "="*100)
    print("REFINEMENT COMPLETE")
    print("="*100)
    print(f"\nTrain images:")
    print(f"  Successful: {train_successful}/{len(train_ids)} ({train_successful/len(train_ids)*100:.1f}%)")
    print(f"  Failed:     {train_failed}")
    print(f"\nTest images:")
    print(f"  Successful: {test_successful}/{len(test_ids)} ({test_successful/len(test_ids)*100:.1f}%)")
    print(f"  Failed:     {test_failed}")
    print(f"\nTotal:")
    print(f"  Successful: {train_successful + test_successful}/{len(train_ids) + len(test_ids)}")
    print(f"  Failed:     {train_failed + test_failed}")
    print("="*100)


def generate_raw_refined_predictions():
    """
    Generate RAW refined predictions (no validation cutoff) using Similarity thresh=1.0.

    Outputs predictions that haven't been fused with rough GPS yet.
    These will be used to test different fusion strategies.

    Writes to:
      - refine_matching/train_refined_raw.csv
      - refine_matching/test_refined_raw.csv
    """
    import csv

    print("="*100)
    print("GENERATING RAW REFINED PREDICTIONS")
    print("Method: Similarity thresh=1.0")
    print("No validation cutoff applied - outputting raw refined positions")
    print("="*100)

    root_dir = Path(__file__).parent.parent
    match_dir_root = Path(__file__).parent / "match_visualizations"

    # Load rough predictions to know which images exist
    train_rough_path = root_dir / "rough_matching" / "train_predictions.csv"
    test_rough_path = root_dir / "rough_matching" / "test_predicted.csv"

    def load_image_ids(csv_path):
        """Load image IDs from CSV."""
        ids = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(int(row['id']))
        return sorted(ids)

    train_ids = load_image_ids(train_rough_path)
    test_ids = load_image_ids(test_rough_path)

    print(f"\nFound {len(train_ids)} train images")
    print(f"Found {len(test_ids)} test images")
    print(f"Total: {len(train_ids) + len(test_ids)} images\n")

    # Process train images
    train_results = []
    train_successful = 0
    train_failed = 0

    print("Processing train images...")
    for img_id in tqdm(train_ids, desc="Train images", unit="img"):
        match_file = match_dir_root / f"img_{img_id:04d}" / "master_matches.npz"

        if not match_file.exists():
            # No match data - will output None for this image
            train_results.append({'id': img_id, 'x_pixel': None, 'y_pixel': None, 'status': 'no_match_data'})
            train_failed += 1
            continue

        # Load match data
        data = np.load(match_file)
        kp_drone = data['kp_drone']
        kp_crop = data['kp_crop']
        H = data['H']
        mask = data['mask']
        crop_offset = tuple(data['crop_offset'])
        drone_shape = tuple(data['drone_shape'])
        rough_x, rough_y = data['rough_position']

        # Apply Similarity thresh=1.0 method
        x_refined, y_refined, description = method_similarity_ransac_thresh(
            kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=1.0
        )

        if x_refined is None:
            # Method failed - record as None
            train_results.append({'id': img_id, 'x_pixel': None, 'y_pixel': None, 'status': 'method_failed'})
            train_failed += 1
        else:
            # Calculate movement from rough GPS
            movement = np.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)

            # Record the raw refined position (no cutoff applied)
            train_results.append({
                'id': img_id,
                'x_pixel': x_refined,
                'y_pixel': y_refined,
                'status': 'success',
                'movement': movement
            })
            train_successful += 1

    # Process test images
    test_results = []
    test_successful = 0
    test_failed = 0

    print("\nProcessing test images...")
    for img_id in tqdm(test_ids, desc="Test images", unit="img"):
        match_file = match_dir_root / f"img_{img_id:04d}" / "master_matches.npz"

        if not match_file.exists():
            test_results.append({'id': img_id, 'x_pixel': None, 'y_pixel': None, 'status': 'no_match_data'})
            test_failed += 1
            continue

        # Load match data
        data = np.load(match_file)
        kp_drone = data['kp_drone']
        kp_crop = data['kp_crop']
        H = data['H']
        mask = data['mask']
        crop_offset = tuple(data['crop_offset'])
        drone_shape = tuple(data['drone_shape'])
        rough_x, rough_y = data['rough_position']

        # Apply Similarity thresh=1.0 method
        x_refined, y_refined, description = method_similarity_ransac_thresh(
            kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=1.0
        )

        if x_refined is None:
            test_results.append({'id': img_id, 'x_pixel': None, 'y_pixel': None, 'status': 'method_failed'})
            test_failed += 1
        else:
            movement = np.sqrt((x_refined - rough_x)**2 + (y_refined - rough_y)**2)
            test_results.append({
                'id': img_id,
                'x_pixel': x_refined,
                'y_pixel': y_refined,
                'status': 'success',
                'movement': movement
            })
            test_successful += 1

    # Write train predictions (CSV format - only id, x_pixel, y_pixel)
    train_output = Path(__file__).parent / "train_refined_raw.csv"
    with open(train_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        for row in train_results:
            writer.writerow({'id': row['id'], 'x_pixel': row['x_pixel'], 'y_pixel': row['y_pixel']})

    print(f"\n✓ Wrote raw train predictions to: {train_output}")

    # Write test predictions
    test_output = Path(__file__).parent / "test_refined_raw.csv"
    with open(test_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'x_pixel', 'y_pixel'])
        writer.writeheader()
        for row in test_results:
            writer.writerow({'id': row['id'], 'x_pixel': row['x_pixel'], 'y_pixel': row['y_pixel']})

    print(f"✓ Wrote raw test predictions to: {test_output}")

    # Summary
    print("\n" + "="*100)
    print("RAW REFINEMENT COMPLETE")
    print("="*100)
    print(f"\nTrain images:")
    print(f"  Successful: {train_successful}/{len(train_ids)} ({train_successful/len(train_ids)*100:.1f}%)")
    print(f"  Failed:     {train_failed}")
    print(f"\nTest images:")
    print(f"  Successful: {test_successful}/{len(test_ids)} ({test_successful/len(test_ids)*100:.1f}%)")
    print(f"  Failed:     {test_failed}")
    print(f"\nTotal:")
    print(f"  Successful: {train_successful + test_successful}/{len(train_ids) + len(test_ids)}")
    print(f"  Failed:     {train_failed + test_failed}")
    print("\nNext step: Run test_fusion_strategies.py to find optimal fusion approach")
    print("="*100)


if __name__ == "__main__":
    import sys

    # Check for flags
    if len(sys.argv) > 1 and sys.argv[1] == "--refine":
        generate_refined_predictions()
    elif len(sys.argv) > 1 and sys.argv[1] == "--raw":
        generate_raw_refined_predictions()
    else:
        main()
