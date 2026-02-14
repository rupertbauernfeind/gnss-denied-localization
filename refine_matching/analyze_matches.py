#!/usr/bin/env python3
"""
Analyze saved match data and try different position calculation methods.

Usage:
    python analyze_matches.py [image_id]
"""

import sys
import csv
import math
from pathlib import Path
import numpy as np
import cv2


def load_csv(path):
    """Load CSV file with id, x_pixel, y_pixel columns."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[int(row["id"])] = (float(row["x_pixel"]), float(row["y_pixel"]))
    return data


def method_homography_center(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Original method: transform center of drone image through homography."""
    h_drone, w_drone = drone_shape
    center = np.float32([[w_drone / 2, h_drone / 2]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]
    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, "Homography on center point"


def method_median_transform(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Transform all inlier keypoints and take median of resulting positions."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Median transform (no inliers)"

    # Transform each drone keypoint to crop space
    transformed = cv2.perspectiveTransform(
        inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    # Take median of transformed positions
    median_x = np.median(transformed[:, 0])
    median_y = np.median(transformed[:, 1])

    x_map = median_x + crop_offset[0]
    y_map = median_y + crop_offset[1]
    return x_map, y_map, f"Median of {len(inlier_kp_drone)} transformed inliers"


def method_mean_transform(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Transform all inlier keypoints and take mean of resulting positions."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Mean transform (no inliers)"

    # Transform each drone keypoint to crop space
    transformed = cv2.perspectiveTransform(
        inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    # Take mean of transformed positions
    mean_x = np.mean(transformed[:, 0])
    mean_y = np.mean(transformed[:, 1])

    x_map = mean_x + crop_offset[0]
    y_map = mean_y + crop_offset[1]
    return x_map, y_map, f"Mean of {len(inlier_kp_drone)} transformed inliers"


def method_median_displacement(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Calculate median displacement vector from drone to crop keypoints."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Median displacement (no inliers)"

    # Calculate displacement for each match
    displacements = inlier_kp_crop - inlier_kp_drone

    # Take median displacement
    median_dx = np.median(displacements[:, 0])
    median_dy = np.median(displacements[:, 1])

    # Apply to center of drone image
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    x_crop = center_x + median_dx
    y_crop = center_y + median_dy

    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Median displacement from {len(inlier_kp_drone)} inliers"


def method_mean_displacement(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Calculate mean displacement vector from drone to crop keypoints."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Mean displacement (no inliers)"

    # Calculate displacement for each match
    displacements = inlier_kp_crop - inlier_kp_drone

    # Take mean displacement
    mean_dx = np.mean(displacements[:, 0])
    mean_dy = np.mean(displacements[:, 1])

    # Apply to center of drone image
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    x_crop = center_x + mean_dx
    y_crop = center_y + mean_dy

    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Mean displacement from {len(inlier_kp_drone)} inliers"


def method_homography_strict_ransac(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Recompute homography with stricter RANSAC threshold."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, "Strict RANSAC (not enough inliers)"

    # Recompute with stricter threshold (2.0 instead of 5.0)
    src_pts = inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = inlier_kp_crop.reshape(-1, 1, 2).astype(np.float32)
    H_strict, mask_strict = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if H_strict is None:
        return None, None, "Strict RANSAC (failed)"

    n_inliers_strict = int(mask_strict.sum()) if mask_strict is not None else 0

    # Transform center
    h_drone, w_drone = drone_shape
    center = np.float32([[w_drone / 2, h_drone / 2]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_strict)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Strict RANSAC (thresh=2.0, {n_inliers_strict} inliers)"


def method_homography_corners(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Transform all 4 corners of drone image and take their centroid."""
    h_drone, w_drone = drone_shape
    corners = np.float32([
        [0, 0],
        [w_drone, 0],
        [w_drone, h_drone],
        [0, h_drone]
    ]).reshape(-1, 1, 2)

    # Transform corners
    transformed_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # Take centroid
    centroid_x = np.mean(transformed_corners[:, 0])
    centroid_y = np.mean(transformed_corners[:, 1])

    x_map = centroid_x + crop_offset[0]
    y_map = centroid_y + crop_offset[1]
    return x_map, y_map, "Centroid of 4 transformed corners"


def method_direct_correspondence(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Find crop position corresponding to drone center via nearest match."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Direct correspondence (no inliers)"

    # Find closest keypoint to drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )
    closest_idx = np.argmin(distances)

    # Use its corresponding crop position
    x_crop = inlier_kp_crop[closest_idx, 0]
    y_crop = inlier_kp_crop[closest_idx, 1]

    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]

    dist = distances[closest_idx]
    return x_map, y_map, f"Nearest match to center (dist={dist:.1f}px)"


def method_weighted_by_distance(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Weight crop positions by inverse distance from drone center."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Distance-weighted (no inliers)"

    # Calculate distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Weight by inverse distance (add small epsilon to avoid division by zero)
    weights = 1.0 / (distances + 1.0)
    weights = weights / weights.sum()  # Normalize

    # Weighted average of transformed positions
    transformed = cv2.perspectiveTransform(
        inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    weighted_x = np.sum(transformed[:, 0] * weights)
    weighted_y = np.sum(transformed[:, 1] * weights)

    x_map = weighted_x + crop_offset[0]
    y_map = weighted_y + crop_offset[1]
    return x_map, y_map, f"Distance-weighted average of {len(inlier_kp_drone)} inliers"


def method_affine_ransac(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Estimate affine transform (6 DOF) with RANSAC."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 3:
        return None, None, "Affine RANSAC (not enough inliers)"

    # Estimate affine transform (6 DOF instead of homography's 8 DOF)
    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                       ransacReprojThreshold=2.0)

    if M is None:
        return None, None, "Affine RANSAC (failed)"

    n_inliers_affine = int(inliers.sum()) if inliers is not None else 0

    # Transform center using affine
    h_drone, w_drone = drone_shape
    center = np.array([[w_drone / 2, h_drone / 2]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Affine RANSAC (thresh=2.0, {n_inliers_affine} inliers)"


def method_similarity_ransac(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Estimate similarity transform (4 DOF: rotation, scale, translation)."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 2:
        return None, None, "Similarity RANSAC (not enough inliers)"

    # Estimate partial affine (similarity: rotation + uniform scale + translation)
    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=2.0)

    if M is None:
        return None, None, "Similarity RANSAC (failed)"

    n_inliers_sim = int(inliers.sum()) if inliers is not None else 0

    # Transform center using similarity
    h_drone, w_drone = drone_shape
    center = np.array([[w_drone / 2, h_drone / 2]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Similarity RANSAC (4-DOF, {n_inliers_sim} inliers)"


def method_affine_median(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Affine transform + median of transformed inliers."""
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 3:
        return None, None, "Affine median (not enough inliers)"

    # Estimate affine transform
    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M, inliers_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                            ransacReprojThreshold=2.0)

    if M is None or inliers_mask is None:
        return None, None, "Affine median (failed)"

    # Use only affine inliers
    affine_inliers = inliers_mask.ravel().astype(bool)
    affine_kp_drone = src_pts[affine_inliers]

    if len(affine_kp_drone) == 0:
        return None, None, "Affine median (no inliers)"

    # Transform all affine inliers and take median
    transformed = cv2.transform(affine_kp_drone.reshape(-1, 1, 2), M).reshape(-1, 2)

    median_x = np.median(transformed[:, 0])
    median_y = np.median(transformed[:, 1])

    x_map = median_x + crop_offset[0]
    y_map = median_y + crop_offset[1]
    return x_map, y_map, f"Affine median ({len(affine_kp_drone)} inliers)"


def method_weighted_displacement(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Weighted median displacement by distance from center."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Weighted displacement (no inliers)"

    # Calculate displacements
    displacements = inlier_kp_crop - inlier_kp_drone

    # Calculate distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Weight by inverse distance
    weights = 1.0 / (distances + 1.0)
    weights = weights / weights.sum()

    # Weighted median approximation (weighted mean for simplicity)
    weighted_dx = np.sum(displacements[:, 0] * weights)
    weighted_dy = np.sum(displacements[:, 1] * weights)

    x_crop = center_x + weighted_dx
    y_crop = center_y + weighted_dy

    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Weighted displacement from {len(inlier_kp_drone)} inliers"


def method_top_k_nearest(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, k=5):
    """Average of K nearest matches to center."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, f"Top-{k} nearest (no inliers)"

    # Find distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Get K nearest
    k_actual = min(k, len(distances))
    nearest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]

    # Average their crop positions
    avg_x = np.mean(inlier_kp_crop[nearest_indices, 0])
    avg_y = np.mean(inlier_kp_crop[nearest_indices, 1])

    x_map = avg_x + crop_offset[0]
    y_map = avg_y + crop_offset[1]
    return x_map, y_map, f"Top-{k_actual} nearest average"


def method_median_k_nearest(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, k=10):
    """Median of K nearest matches to center."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, f"Median top-{k} (no inliers)"

    # Find distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Get K nearest
    k_actual = min(k, len(distances))
    nearest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]

    # Median of their crop positions
    median_x = np.median(inlier_kp_crop[nearest_indices, 0])
    median_y = np.median(inlier_kp_crop[nearest_indices, 1])

    x_map = median_x + crop_offset[0]
    y_map = median_y + crop_offset[1]
    return x_map, y_map, f"Median of top-{k_actual} nearest"


def method_quadratic_weighted(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Quadratic inverse distance weighting (more emphasis on closest)."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Quadratic weighted (no inliers)"

    # Calculate distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Quadratic inverse distance weighting
    weights = 1.0 / ((distances + 1.0) ** 2)
    weights = weights / weights.sum()

    # Weighted average
    weighted_x = np.sum(inlier_kp_crop[:, 0] * weights)
    weighted_y = np.sum(inlier_kp_crop[:, 1] * weights)

    x_map = weighted_x + crop_offset[0]
    y_map = weighted_y + crop_offset[1]
    return x_map, y_map, f"Quadratic weighted ({len(inlier_kp_drone)} inliers)"


def method_exponential_weighted(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Exponential distance weighting (strong emphasis on closest)."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, "Exponential weighted (no inliers)"

    # Calculate distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Exponential weighting (sigma = 50 pixels)
    weights = np.exp(-distances / 50.0)
    weights = weights / weights.sum()

    # Weighted average
    weighted_x = np.sum(inlier_kp_crop[:, 0] * weights)
    weighted_y = np.sum(inlier_kp_crop[:, 1] * weights)

    x_map = weighted_x + crop_offset[0]
    y_map = weighted_y + crop_offset[1]
    return x_map, y_map, f"Exponential weighted ({len(inlier_kp_drone)} inliers)"


def method_radius_filter(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, radius=100):
    """Only use matches within radius pixels of center."""
    h_drone, w_drone = drone_shape
    center_x = w_drone / 2
    center_y = h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, f"Radius filter r={radius} (no inliers)"

    # Calculate distances from drone center
    distances = np.sqrt(
        (inlier_kp_drone[:, 0] - center_x) ** 2 +
        (inlier_kp_drone[:, 1] - center_y) ** 2
    )

    # Filter by radius
    within_radius = distances <= radius
    if not within_radius.any():
        return None, None, f"Radius filter r={radius} (no matches within radius)"

    # Median of filtered matches
    filtered_crop = inlier_kp_crop[within_radius]
    median_x = np.median(filtered_crop[:, 0])
    median_y = np.median(filtered_crop[:, 1])

    x_map = median_x + crop_offset[0]
    y_map = median_y + crop_offset[1]
    n_filtered = int(within_radius.sum())
    return x_map, y_map, f"Radius filter r={radius} ({n_filtered}/{len(inlier_kp_drone)} inliers)"


def method_local_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, radius=100):
    """Recompute homography using only matches within radius of drone center."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) == 0:
        return None, None, f"Local H r={radius} (no inliers)"

    distances = np.sqrt((inlier_kp_drone[:, 0] - center_x)**2 + (inlier_kp_drone[:, 1] - center_y)**2)
    within_radius = distances <= radius
    local_kp_drone = inlier_kp_drone[within_radius]
    local_kp_crop = inlier_kp_crop[within_radius]

    if len(local_kp_drone) < 4:
        return None, None, f"Local H r={radius} ({len(local_kp_drone)} matches, need 4+)"

    src_pts = local_kp_drone.reshape(-1, 1, 2).astype(np.float32)
    dst_pts = local_kp_crop.reshape(-1, 1, 2).astype(np.float32)
    H_local, mask_local = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)

    if H_local is None:
        return None, None, f"Local H r={radius} (RANSAC failed)"

    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_local)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Local H r={radius}"


def method_weighted_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, sigma=100):
    """Compute H using weighted DLT with Gaussian distance weighting."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, f"Weighted H σ={sigma} (insufficient inliers)"

    distances = np.sqrt((inlier_kp_drone[:, 0] - center_x)**2 + (inlier_kp_drone[:, 1] - center_y)**2)
    weights = np.exp(-distances**2 / (2 * sigma**2))

    # Build weighted DLT matrix
    n = len(inlier_kp_drone)
    A = np.zeros((2*n, 9))
    for i in range(n):
        x, y = inlier_kp_drone[i]
        xp, yp = inlier_kp_crop[i]
        w = np.sqrt(weights[i])
        A[2*i] = w * np.array([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A[2*i+1] = w * np.array([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

    # Solve via SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_weighted = h.reshape(3, 3) / h.reshape(3, 3)[2, 2]

    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_weighted)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Weighted H σ={sigma}"


def method_homography_decomposition(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, focal_length_px=None):
    """Decompose homography to recover camera pose and compute true camera position.

    Uses cv2.decomposeHomographyMat to extract camera rotation and translation from H.
    The homography relates a planar scene (satellite map at Z=0) to the camera image.

    Args:
        focal_length_px: Estimated focal length in pixels (if None, uses typical drone camera ~0.7*width)
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape

    # Estimate camera intrinsics K
    # For typical drone cameras: focal length is ~0.7-0.8 of image width
    if focal_length_px is None:
        focal_length_px = 0.75 * w_drone  # Conservative estimate

    cx = w_drone / 2  # Principal point x
    cy = h_drone / 2  # Principal point y

    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Decompose homography to get camera pose
    # Returns up to 4 solutions: rotations, translations, normals
    try:
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    except cv2.error as e:
        return None, None, f"H decompose (decomposition failed: {e})"

    if num_solutions == 0:
        return None, None, "H decompose (no solutions)"

    # Select the physically valid solution:
    # - Camera should be above ground (t_z > 0)
    # - Normal should point toward camera (n_z > 0)
    # - Positive depth for observed points

    best_solution = None
    best_score = -np.inf

    for i in range(num_solutions):
        R = rotations[i]
        t = translations[i].flatten()
        n = normals[i].flatten()

        # Check if solution is physically plausible
        # Camera height (t_z) should be positive
        if t[2] <= 0:
            continue

        # Normal should point upward (toward camera)
        if n[2] <= 0:
            continue

        # Test if matches have positive depth when projected
        # For a valid solution, points should be in front of the camera
        inlier_mask = mask.ravel().astype(bool)
        inlier_kp_drone = kp_drone[inlier_mask]
        inlier_kp_crop = kp_crop[inlier_mask]

        # Sample a few points to test depth
        sample_size = min(10, len(inlier_kp_crop))
        sample_indices = np.random.choice(len(inlier_kp_crop), sample_size, replace=False)

        positive_depths = 0
        for idx in sample_indices:
            # Map point in world coordinates (Z=0 plane)
            X_world = np.array([inlier_kp_crop[idx, 0], inlier_kp_crop[idx, 1], 0, 1])

            # Transform to camera coordinates: X_cam = R * X_world + t
            X_world_3d = X_world[:3]
            X_cam = R @ X_world_3d + t

            # Depth is the Z coordinate in camera frame
            if X_cam[2] > 0:
                positive_depths += 1

        # Score based on positive depths and camera height
        score = positive_depths + min(t[2] / 100, 10)  # Reward reasonable height

        if score > best_score:
            best_score = score
            best_solution = (R, t, n)

    if best_solution is None:
        return None, None, "H decompose (no valid solution)"

    R, t, n = best_solution

    # The camera center in world coordinates is: C = -R^T @ t
    # But we need to be careful about coordinate systems
    #
    # The homography maps image -> world plane (satellite map)
    # H = K [r1 r2 t] where r1, r2 are first two columns of R
    #
    # For a point on the ground plane (Z=0):
    # x_cam = K * [R|t] * [X, Y, 0, 1]^T
    #
    # The camera center in map coordinates:
    camera_center_map = -R.T @ t

    # Extract X, Y position in satellite map coordinates
    x_map = camera_center_map[0] + crop_offset[0]
    y_map = camera_center_map[1] + crop_offset[1]

    return x_map, y_map, f"H decompose (f={focal_length_px:.0f}px)"


def method_pnp_solver(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, method_name="RANSAC"):
    """Use OpenCV's solvePnP to estimate camera pose from 2D-3D correspondences.

    Treats satellite map coordinates as ground plane (Z=0) and uses solvePnP
    to find camera position and orientation.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 10:
        return None, None, f"PnP {method_name} (insufficient matches)"

    # 3D world points (map coords with Z=0)
    object_points = np.column_stack([
        inlier_kp_crop[:, 0],  # X
        inlier_kp_crop[:, 1],  # Y
        np.zeros(len(inlier_kp_crop))  # Z=0
    ]).astype(np.float32)

    # 2D image points
    image_points = inlier_kp_drone.astype(np.float32)

    # Real camera intrinsics for original 8000x6000 image
    fx_orig = 6591.648470216358
    fy_orig = 6620.211673955053
    cx_orig = 4000.0
    cy_orig = 3000.0
    w_orig = 8000.0
    h_orig = 6000.0

    # Scale to current image size
    scale_x = w_drone / w_orig
    scale_y = h_drone / h_orig
    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Solve PnP
    try:
        if method_name == "RANSAC":
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, camera_matrix, None,
                reprojectionError=2.0, confidence=0.99
            )
        elif method_name == "ITERATIVE":
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:  # "DLS" or other
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, None,
                flags=cv2.SOLVEPNP_DLS
            )

        if not success:
            return None, None, f"PnP {method_name} (failed)"

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Camera center in world coordinates: C = -R^T @ t
        camera_center = -R.T @ tvec.flatten()

        # Extract X, Y position
        x_map = camera_center[0] + crop_offset[0]
        y_map = camera_center[1] + crop_offset[1]

        return x_map, y_map, f"PnP {method_name}"

    except Exception as e:
        return None, None, f"PnP {method_name} (error: {e})"


def method_pnp_hybrid(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Hybrid: PnP with real intrinsics, validated against homography baseline.

    Uses solvePnP with calibrated intrinsics, but checks if result is reasonable
    by comparing to homography center projection. Falls back if unreasonable.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    cx_img, cy_img = w_drone / 2, h_drone / 2

    # Get baseline from homography
    center = np.float32([[cx_img, cy_img]]).reshape(-1, 1, 2)
    proj_center = cv2.perspectiveTransform(center, H)
    x_baseline = proj_center[0, 0, 0] + crop_offset[0]
    y_baseline = proj_center[0, 0, 1] + crop_offset[1]

    # Try PnP
    x_pnp, y_pnp, status = method_pnp_solver(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, "RANSAC")

    # If PnP failed or gives result too far from baseline, use baseline
    if x_pnp is None:
        return x_baseline, y_baseline, "PnP hybrid (PnP failed, using H)"

    dist = np.sqrt((x_pnp - x_baseline)**2 + (y_pnp - y_baseline)**2)

    # If PnP result is within 100px of homography, trust it
    # Otherwise, use homography (PnP probably failed)
    if dist < 100:
        return x_pnp, y_pnp, f"PnP hybrid (PnP, dist={dist:.0f}px)"
    else:
        return x_baseline, y_baseline, f"PnP hybrid (H, PnP dist={dist:.0f}px)"


def method_pycolmap_ba(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Use PyCOLMAP's absolute pose refinement.

    Uses pycolmap's pose estimation and refinement on 2D-3D correspondences
    with known camera intrinsics.
    """
    try:
        import pycolmap
        import cv2
        import numpy as np
    except ImportError:
        return None, None, "PyCOLMAP BA (not installed)"

    h_drone, w_drone = drone_shape

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 10:
        return None, None, "PyCOLMAP BA (insufficient matches)"

    # Real camera intrinsics scaled to current image size
    fx_orig = 6591.648470216358
    fy_orig = 6620.211673955053
    cx_orig = 4000.0
    cy_orig = 3000.0
    w_orig = 8000.0
    h_orig = 6000.0

    scale_x = w_drone / w_orig
    scale_y = h_drone / h_orig
    fx = fx_orig * scale_x
    fy = fy_orig * scale_y
    cx = cx_orig * scale_x
    cy = cy_orig * scale_y

    try:
        # Prepare 2D-3D correspondences
        points2D = inlier_kp_drone.astype(np.float64)
        points3D = np.column_stack([
            inlier_kp_crop[:, 0],
            inlier_kp_crop[:, 1],
            np.zeros(len(inlier_kp_crop))
        ]).astype(np.float64)

        # Create COLMAP camera
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=int(w_drone),
            height=int(h_drone),
            params=np.array([fx, fy, cx, cy], dtype=np.float64)
        )

        # Get homography baseline for validation
        cx_img, cy_img = w_drone / 2, h_drone / 2
        center = np.float32([[cx_img, cy_img]]).reshape(-1, 1, 2)
        proj_center = cv2.perspectiveTransform(center, H)
        x_baseline = proj_center[0, 0, 0] + crop_offset[0]
        y_baseline = proj_center[0, 0, 1] + crop_offset[1]

        # Try multiple RANSAC thresholds and pick result closest to homography baseline
        # Different images need different thresholds due to varying noise levels
        best_result = None
        best_dist_to_baseline = float('inf')

        for max_error in [1.0, 2.0, 4.0, 8.0]:
            try:
                answer = pycolmap.estimate_absolute_pose(
                    points2D,
                    points3D,
                    camera,
                    estimation_options={"ransac": {"max_error": max_error}}
                )

                if answer["num_inliers"] < 10:
                    continue

                # Extract pose
                cam_from_world = answer["cam_from_world"]
                R = cam_from_world.rotation.matrix()
                tvec = cam_from_world.translation
                camera_center = -R.T @ tvec
                x_map = camera_center[0] + crop_offset[0]
                y_map = camera_center[1] + crop_offset[1]

                # Check distance to homography baseline
                dist = np.sqrt((x_map - x_baseline)**2 + (y_map - y_baseline)**2)

                # Keep result closest to baseline (most consistent with 2D geometry)
                if dist < best_dist_to_baseline:
                    best_dist_to_baseline = dist
                    best_result = (x_map, y_map, answer["num_inliers"], max_error)
            except:
                continue

        if best_result is None:
            return None, None, "PyCOLMAP BA (all thresholds failed)"

        x_map, y_map, n_inliers, thresh = best_result

        # If result is too far from baseline, use baseline instead (like PnP hybrid)
        if best_dist_to_baseline > 100:
            return x_baseline, y_baseline, f"PyCOLMAP BA (fallback to H, dist={best_dist_to_baseline:.0f}px)"

        return x_map, y_map, f"PyCOLMAP BA ({n_inliers} inliers, thresh={thresh}, dist={best_dist_to_baseline:.0f}px)"

    except Exception as e:
        import traceback
        return None, None, f"PyCOLMAP BA (error: {str(e)[:50]})"


def analyze_matcher(match_dir, matcher_name, gt_x, gt_y, rough_x, rough_y, rough_err):
    """Analyze one matcher's results with all position calculation methods."""
    match_file = match_dir / f"{matcher_name}_matches.npz"

    if not match_file.exists():
        print(f"\n{matcher_name.upper()}: No match data found")
        return

    # Load match data
    data = np.load(match_file)
    kp_drone = data['kp_drone']
    kp_crop = data['kp_crop']
    H = data['H']
    mask = data['mask']
    n_inliers = int(data['n_inliers'])
    n_matches = len(kp_drone)  # Total matches before RANSAC filtering

    # Load crop offset and drone shape from NPZ file
    crop_offset = tuple(data['crop_offset'])
    drone_shape = tuple(data['drone_shape'])

    print(f"\n{'='*80}")
    print(f"{matcher_name.upper()}: {n_matches} matches, {n_inliers} inliers")
    print(f"{'='*80}")

    # Try all methods
    methods = [
        method_homography_center,
        method_median_transform,
        method_mean_transform,
        method_weighted_by_distance,
        method_homography_corners,
        method_direct_correspondence,
        lambda *args: method_top_k_nearest(*args, k=3),
        lambda *args: method_top_k_nearest(*args, k=5),
        lambda *args: method_top_k_nearest(*args, k=10),
        lambda *args: method_median_k_nearest(*args, k=5),
        lambda *args: method_median_k_nearest(*args, k=10),
        lambda *args: method_median_k_nearest(*args, k=20),
        method_quadratic_weighted,
        method_exponential_weighted,
        lambda *args: method_radius_filter(*args, radius=50),
        lambda *args: method_radius_filter(*args, radius=100),
        lambda *args: method_radius_filter(*args, radius=150),
        method_median_displacement,
        method_mean_displacement,
        method_weighted_displacement,
        method_affine_ransac,
        method_similarity_ransac,
        method_affine_median,
        method_homography_strict_ransac,
    ]

    results = []
    for method in methods:
        x_map, y_map, description = method(kp_drone, kp_crop, H, mask, crop_offset, drone_shape)

        if x_map is None:
            print(f"  ✗ {description}: Failed")
            continue

        err = math.sqrt((x_map - gt_x) ** 2 + (y_map - gt_y) ** 2)
        delta = rough_err - err
        better = "✓ BETTER" if err < rough_err else "  WORSE"

        results.append({
            'method': description,
            'x': x_map,
            'y': y_map,
            'error': err,
            'delta': delta
        })

        print(f"  {better}  err={err:6.1f}px ({err/5:3.0f}m)  delta={delta:+6.1f}px  |  {description}")

    # Print best method
    if results:
        best = min(results, key=lambda r: r['error'])
        print(f"\n  BEST: {best['method']}")
        print(f"        Position: ({best['x']:.1f}, {best['y']:.1f})")
        print(f"        Error: {best['error']:.1f}px ({best['error']/5:.0f}m)")
        print(f"        Improvement: {best['delta']:.1f}px over rough GPS")


def main():
    root = Path(__file__).parent
    repo_root = root.parent

    # Load ground truth and rough predictions
    rough = load_csv(repo_root / "rough_matching" / "train_predictions.csv")
    gt = load_csv(repo_root / "data" / "train_data" / "train_pos.csv")

    # Get image ID from command line or use default
    img_id = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    rough_x, rough_y = rough[img_id]
    gt_x, gt_y = gt[img_id]
    rough_err = math.sqrt((rough_x - gt_x) ** 2 + (rough_y - gt_y) ** 2)

    print(f"Image {img_id}")
    print(f"  GT:    ({gt_x:.1f}, {gt_y:.1f})")
    print(f"  Rough: ({rough_x:.1f}, {rough_y:.1f})  err={rough_err:.1f}px ({rough_err/5:.0f}m)")

    # Find match directory
    match_dir = root / "match_visualizations" / f"img_{img_id:04d}"
    if not match_dir.exists():
        print(f"\nError: No match data found at {match_dir}")
        print(f"Run 'python test_one.py {img_id}' first to generate match data")
        return

    # Analyze Master matcher only
    analyze_matcher(match_dir, 'master', gt_x, gt_y, rough_x, rough_y, rough_err)


def method_reproj_filtered_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=2.0):
    """Filter outliers by reprojection error, then recompute homography.

    Well-known technique: Remove matches with high reprojection error under
    the initial homography, then recompute H with the filtered set for better accuracy.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, f"Reproj filter thresh={threshold} (no inliers)"

    # Compute reprojection errors for all inliers
    src_pts = inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32)
    proj_pts = cv2.perspectiveTransform(src_pts, H)

    # Reprojection error
    errors = np.sqrt(np.sum((proj_pts[:, 0, :] - inlier_kp_crop)**2, axis=1))

    # Keep only matches with low reprojection error
    good_mask = errors < threshold
    filtered_drone = inlier_kp_drone[good_mask]
    filtered_crop = inlier_kp_crop[good_mask]

    if len(filtered_drone) < 4:
        return None, None, f"Reproj filter thresh={threshold} ({len(filtered_drone)} left, need 4+)"

    # Recompute homography with filtered matches
    src_filtered = filtered_drone.reshape(-1, 1, 2).astype(np.float32)
    dst_filtered = filtered_crop.reshape(-1, 1, 2).astype(np.float32)
    H_filtered, _ = cv2.findHomography(src_filtered, dst_filtered, 0)  # Use all filtered points

    if H_filtered is None:
        return None, None, f"Reproj filter thresh={threshold} (H computation failed)"

    # Project center
    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_filtered)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    pct = 100 * len(filtered_drone) / len(inlier_kp_drone)
    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Reproj filter thresh={threshold} ({pct:.0f}% kept)"


def method_normalized_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Compute homography using normalized DLT for better numerical conditioning.

    Well-known technique from Hartley & Zisserman: Normalize point coordinates
    to have zero mean and unit variance before computing homography for better
    numerical stability.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, "Normalized H (no inliers)"

    # Normalize drone points: T1
    mean_drone = inlier_kp_drone.mean(axis=0)
    std_drone = inlier_kp_drone.std()
    if std_drone < 1e-6:
        return None, None, "Normalized H (zero variance)"

    scale_drone = np.sqrt(2) / std_drone
    T1 = np.array([
        [scale_drone, 0, -scale_drone * mean_drone[0]],
        [0, scale_drone, -scale_drone * mean_drone[1]],
        [0, 0, 1]
    ])

    # Normalize crop points: T2
    mean_crop = inlier_kp_crop.mean(axis=0)
    std_crop = inlier_kp_crop.std()
    if std_crop < 1e-6:
        return None, None, "Normalized H (zero variance)"

    scale_crop = np.sqrt(2) / std_crop
    T2 = np.array([
        [scale_crop, 0, -scale_crop * mean_crop[0]],
        [0, scale_crop, -scale_crop * mean_crop[1]],
        [0, 0, 1]
    ])

    # Apply normalization
    ones = np.ones((len(inlier_kp_drone), 1))
    drone_hom = np.hstack([inlier_kp_drone, ones])
    crop_hom = np.hstack([inlier_kp_crop, ones])

    drone_norm = (T1 @ drone_hom.T).T
    crop_norm = (T2 @ crop_hom.T).T

    # Compute normalized homography
    src_norm = drone_norm[:, :2].reshape(-1, 1, 2).astype(np.float32)
    dst_norm = crop_norm[:, :2].reshape(-1, 1, 2).astype(np.float32)
    H_norm, _ = cv2.findHomography(src_norm, dst_norm, cv2.RANSAC, 2.0)

    if H_norm is None:
        return None, None, "Normalized H (RANSAC failed)"

    # Denormalize: H = T2^-1 @ H_norm @ T1
    H_denorm = np.linalg.inv(T2) @ H_norm @ T1

    # Project center
    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_denorm)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    return x_crop + crop_offset[0], y_crop + crop_offset[1], "Normalized H"


def method_irls_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, n_iterations=5):
    """Iteratively Reweighted Least Squares homography with Huber weighting.

    Well-known robust estimation technique: Iteratively downweight outliers
    using Huber M-estimator for robust homography estimation.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, "IRLS H (no inliers)"

    # Start with uniform weights
    n = len(inlier_kp_drone)
    weights = np.ones(n)

    H_current = H.copy()

    for iteration in range(n_iterations):
        # Build weighted DLT matrix
        A = np.zeros((2*n, 9))
        for i in range(n):
            x, y = inlier_kp_drone[i]
            xp, yp = inlier_kp_crop[i]
            w = np.sqrt(weights[i])
            A[2*i] = w * np.array([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A[2*i+1] = w * np.array([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

        # Solve via SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1]
        H_current = h.reshape(3, 3)
        H_current = H_current / H_current[2, 2]

        # Compute residuals
        src_pts = inlier_kp_drone.reshape(-1, 1, 2).astype(np.float32)
        proj_pts = cv2.perspectiveTransform(src_pts, H_current)
        residuals = np.sqrt(np.sum((proj_pts[:, 0, :] - inlier_kp_crop)**2, axis=1))

        # Huber weighting: w = 1 for small errors, w = k/|r| for large errors
        k = 1.345 * np.median(residuals)  # Huber threshold
        weights = np.where(residuals <= k, 1.0, k / residuals)

    # Project center
    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_current)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"IRLS H ({n_iterations} iter)"


def method_local_weighted_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, radius=100, sigma=75):
    """Combine local filtering AND Gaussian weighting for center-focused homography.

    Hybrid approach: First filter to local region, then apply Gaussian weights
    within that region for doubly-focused optimization.
    """
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, f"Local+Weighted H r={radius},σ={sigma} (no inliers)"

    # First: spatial filtering
    distances = np.sqrt((inlier_kp_drone[:, 0] - center_x)**2 + (inlier_kp_drone[:, 1] - center_y)**2)
    within_radius = distances <= radius
    local_kp_drone = inlier_kp_drone[within_radius]
    local_kp_crop = inlier_kp_crop[within_radius]
    local_distances = distances[within_radius]

    if len(local_kp_drone) < 4:
        return None, None, f"Local+Weighted H r={radius},σ={sigma} ({len(local_kp_drone)} matches, need 4+)"

    # Second: Gaussian weighting on filtered points
    weights = np.exp(-local_distances**2 / (2 * sigma**2))

    # Build weighted DLT matrix
    n = len(local_kp_drone)
    A = np.zeros((2*n, 9))
    for i in range(n):
        x, y = local_kp_drone[i]
        xp, yp = local_kp_crop[i]
        w = np.sqrt(weights[i])
        A[2*i] = w * np.array([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A[2*i+1] = w * np.array([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

    # Solve via SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_combined = h.reshape(3, 3)
    H_combined = H_combined / H_combined[2, 2]

    # Project center
    center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(center, H_combined)
    x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Local+Weighted H r={radius},σ={sigma}"


def method_symmetric_transfer_homography(kp_drone, kp_crop, H, mask, crop_offset, drone_shape):
    """Recompute homography minimizing symmetric transfer error instead of one-way.

    Well-known technique: Minimize d(H*x, x')^2 + d(H^-1*x', x)^2 instead of
    just forward reprojection error for more symmetric, robust estimation.
    """
    import cv2
    import numpy as np
    from scipy.optimize import least_squares

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 4:
        return None, None, "Symmetric H (no inliers)"

    # Initialize with standard homography
    h_init = H.flatten()[:8]  # 8 DOF (h33 = 1)

    def symmetric_error(h_params):
        """Compute symmetric transfer error."""
        h = np.append(h_params, 1.0)
        H_mat = h.reshape(3, 3)

        try:
            H_inv = np.linalg.inv(H_mat)
        except:
            return np.ones(2 * len(inlier_kp_drone)) * 1e6

        errors = []
        for i in range(len(inlier_kp_drone)):
            x = np.array([inlier_kp_drone[i, 0], inlier_kp_drone[i, 1], 1])
            xp = np.array([inlier_kp_crop[i, 0], inlier_kp_crop[i, 1], 1])

            # Forward: H*x -> x'
            proj_forward = H_mat @ x
            proj_forward /= proj_forward[2]
            err_forward = np.linalg.norm(proj_forward[:2] - xp[:2])

            # Backward: H^-1*x' -> x
            proj_backward = H_inv @ xp
            proj_backward /= proj_backward[2]
            err_backward = np.linalg.norm(proj_backward[:2] - x[:2])

            errors.append(err_forward)
            errors.append(err_backward)

        return np.array(errors)

    try:
        # Optimize symmetric transfer error
        result = least_squares(symmetric_error, h_init, max_nfev=50, ftol=1e-6)
        h_opt = np.append(result.x, 1.0)
        H_sym = h_opt.reshape(3, 3)

        # Project center
        center = np.float32([[center_x, center_y]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(center, H_sym)
        x_crop, y_crop = proj[0, 0, 0], proj[0, 0, 1]

        return x_crop + crop_offset[0], y_crop + crop_offset[1], "Symmetric H"
    except:
        return None, None, "Symmetric H (optimization failed)"


def method_similarity_ransac_thresh(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=2.0):
    """Similarity RANSAC with configurable threshold."""
    import cv2
    import numpy as np

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 2:
        return None, None, f"Similarity thresh={threshold} (not enough inliers)"

    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=threshold)

    if M is None:
        return None, None, f"Similarity thresh={threshold} (failed)"

    n_inliers_sim = int(inliers.sum()) if inliers is not None else 0

    h_drone, w_drone = drone_shape
    center = np.array([[w_drone / 2, h_drone / 2]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    x_map = x_crop + crop_offset[0]
    y_map = y_crop + crop_offset[1]
    return x_map, y_map, f"Similarity thresh={threshold} ({n_inliers_sim} inliers)"


def method_reproj_filtered_similarity(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, threshold=2.0):
    """Filter outliers by reprojection error, then recompute similarity transform."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 2:
        return None, None, f"Reproj Sim thresh={threshold} (no inliers)"

    # Initial similarity estimate
    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M_init, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)

    if M_init is None:
        return None, None, f"Reproj Sim thresh={threshold} (initial failed)"

    # Compute reprojection errors
    src_pts_reshaped = src_pts.reshape(-1, 1, 2)
    proj_pts = cv2.transform(src_pts_reshaped, M_init).reshape(-1, 2)
    errors = np.linalg.norm(proj_pts - dst_pts, axis=1)

    # Filter by threshold
    good_mask = errors <= threshold
    if good_mask.sum() < 2:
        return None, None, f"Reproj Sim thresh={threshold} (too few after filter)"

    # Recompute with filtered matches
    filtered_src = src_pts[good_mask]
    filtered_dst = dst_pts[good_mask]
    M_refined, _ = cv2.estimateAffinePartial2D(filtered_src, filtered_dst, method=cv2.LMEDS)

    if M_refined is None:
        # Fallback to initial estimate
        M_refined = M_init

    center = np.array([[w_drone / 2, h_drone / 2]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M_refined).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Reproj Sim thresh={threshold}"


def method_iterative_similarity(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, n_iterations=2):
    """Iteratively refine similarity transform by filtering outliers."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    inlier_mask = mask.ravel().astype(bool)
    current_src = kp_drone[inlier_mask].astype(np.float32)
    current_dst = kp_crop[inlier_mask].astype(np.float32)

    if len(current_src) < 2:
        return None, None, f"Iterative Sim n={n_iterations} (no inliers)"

    M = None
    for i in range(n_iterations):
        M_iter, inliers = cv2.estimateAffinePartial2D(current_src, current_dst,
                                                        method=cv2.RANSAC,
                                                        ransacReprojThreshold=2.0)
        if M_iter is None:
            break

        M = M_iter

        # Filter for next iteration
        if inliers is not None and i < n_iterations - 1:
            inlier_mask_iter = inliers.ravel().astype(bool)
            current_src = current_src[inlier_mask_iter]
            current_dst = current_dst[inlier_mask_iter]

            if len(current_src) < 2:
                break

    if M is None:
        return None, None, f"Iterative Sim n={n_iterations} (failed)"

    center = np.array([[w_drone / 2, h_drone / 2]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Iterative Sim n={n_iterations}"


def method_weighted_similarity(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, sigma=100):
    """Similarity with Gaussian weighting by distance from center."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 2:
        return None, None, f"Weighted Sim σ={sigma} (no inliers)"

    # Initial RANSAC similarity
    src_pts = inlier_kp_drone.astype(np.float32)
    dst_pts = inlier_kp_crop.astype(np.float32)
    M_init, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.0)

    if M_init is None or inliers is None:
        return None, None, f"Weighted Sim σ={sigma} (RANSAC failed)"

    # Use only RANSAC inliers for weighted refinement
    ransac_mask = inliers.ravel().astype(bool)
    ransac_src = src_pts[ransac_mask]
    ransac_dst = dst_pts[ransac_mask]

    if len(ransac_src) < 2:
        return None, None, f"Weighted Sim σ={sigma} (no RANSAC inliers)"

    # Compute weights based on distance from center
    distances = np.sqrt((ransac_src[:, 0] - center_x)**2 + (ransac_src[:, 1] - center_y)**2)
    weights = np.exp(-distances**2 / (2 * sigma**2))

    # Solve weighted least squares for similarity transform
    # Similarity: x' = s*R*x + t  where R is rotation, s is scale, t is translation
    # This is equivalent to: x' = a*x - b*y + tx, y' = b*x + a*y + ty
    # We solve for [a, b, tx, ty]

    n = len(ransac_src)
    A = np.zeros((2*n, 4))
    b_vec = np.zeros(2*n)

    for i in range(n):
        w = np.sqrt(weights[i])
        x, y = ransac_src[i]
        xp, yp = ransac_dst[i]

        A[2*i] = w * np.array([x, -y, 1, 0])
        A[2*i+1] = w * np.array([y, x, 0, 1])
        b_vec[2*i] = w * xp
        b_vec[2*i+1] = w * yp

    # Solve weighted least squares
    params, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, tx, ty = params

    # Construct similarity matrix
    M_weighted = np.array([[a, -b, tx],
                           [b, a, ty]], dtype=np.float32)

    center = np.array([[center_x, center_y]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M_weighted).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Weighted Sim σ={sigma}"


def method_local_similarity(kp_drone, kp_crop, H, mask, crop_offset, drone_shape, radius=100):
    """Similarity using only matches within radius of drone center."""
    import cv2
    import numpy as np

    h_drone, w_drone = drone_shape
    center_x, center_y = w_drone / 2, h_drone / 2

    inlier_mask = mask.ravel().astype(bool)
    inlier_kp_drone = kp_drone[inlier_mask]
    inlier_kp_crop = kp_crop[inlier_mask]

    if len(inlier_kp_drone) < 2:
        return None, None, f"Local Sim r={radius} (no inliers)"

    # Filter by distance from center
    distances = np.sqrt((inlier_kp_drone[:, 0] - center_x)**2 + (inlier_kp_drone[:, 1] - center_y)**2)
    within_radius = distances <= radius

    if within_radius.sum() < 2:
        return None, None, f"Local Sim r={radius} ({within_radius.sum()} matches, need 2+)"

    local_src = inlier_kp_drone[within_radius].astype(np.float32)
    local_dst = inlier_kp_crop[within_radius].astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(local_src, local_dst, method=cv2.RANSAC,
                                        ransacReprojThreshold=2.0)

    if M is None:
        return None, None, f"Local Sim r={radius} (RANSAC failed)"

    center = np.array([[center_x, center_y]], dtype=np.float32)
    center_transformed = cv2.transform(center.reshape(-1, 1, 2), M).reshape(-1, 2)

    x_crop, y_crop = center_transformed[0]
    return x_crop + crop_offset[0], y_crop + crop_offset[1], f"Local Sim r={radius}"


if __name__ == "__main__":
    main()
