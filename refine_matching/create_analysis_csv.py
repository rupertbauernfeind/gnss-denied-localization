#!/usr/bin/env python3
"""
Create a comprehensive CSV combining predictions, ground truth, and camera parameters.

For each train image, outputs:
- Image ID
- Predicted x, y
- Ground truth x, y
- Camera rotation (as roll, pitch, yaw angles in degrees)
- Camera rotation matrix (9 values)
- Distortion coefficients (6 values)
- Other camera parameters (focal length, principal point, etc.)
"""

import csv
import numpy as np
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def load_csv(path):
    """Load CSV file into dict mapping image_id -> (x, y)."""
    result = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['id'])
            x = float(row['x_pixel'])
            y = float(row['y_pixel'])
            result[img_id] = (x, y)
    return result


def read_camera_params_from_xmp(xmp_path):
    """
    Read camera parameters from XMP file.

    Returns dict with:
    - rotation_matrix: 3x3 numpy array
    - roll_deg, pitch_deg, yaw_deg: Euler angles in degrees
    - distortion_coeffs: list of 6 floats
    - focal_length_35mm: float
    - principal_point_u, principal_point_v: floats
    - skew, aspect_ratio: floats
    """
    if not xmp_path.exists():
        return None

    try:
        tree = ET.parse(xmp_path)
        root = tree.getroot()

        # Navigate to Description element
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'
        }

        desc = root.find('.//rdf:Description', ns)
        if desc is None:
            return None

        # Extract rotation matrix
        rotation_elem = desc.find('xcr:Rotation', ns)
        if rotation_elem is not None:
            rotation_values = [float(x) for x in rotation_elem.text.split()]
            rotation_matrix = np.array(rotation_values).reshape(3, 3)
        else:
            rotation_matrix = None

        # Extract distortion coefficients
        distortion_elem = desc.find('xcr:DistortionCoeficients', ns)
        if distortion_elem is not None:
            distortion_coeffs = [float(x) for x in distortion_elem.text.split()]
        else:
            distortion_coeffs = [0] * 6

        # Extract other parameters from attributes
        focal_length = float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}FocalLength35mm', 0))
        principal_u = float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointU', 0))
        principal_v = float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointV', 0))
        skew = float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}Skew', 0))
        aspect_ratio = float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}AspectRatio', 1))

        # Compute Euler angles from rotation matrix
        if rotation_matrix is not None:
            # Roll (rotation around x-axis)
            roll_rad = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            roll_deg = math.degrees(roll_rad)

            # Pitch (rotation around y-axis)
            pitch_rad = math.atan2(-rotation_matrix[2, 0],
                                   math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            pitch_deg = math.degrees(pitch_rad)

            # Yaw (rotation around z-axis)
            yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            yaw_deg = math.degrees(yaw_rad)
        else:
            roll_deg = pitch_deg = yaw_deg = 0
            rotation_matrix = np.eye(3)

        return {
            'rotation_matrix': rotation_matrix,
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'yaw_deg': yaw_deg,
            'distortion_coeffs': distortion_coeffs,
            'focal_length_35mm': focal_length,
            'principal_point_u': principal_u,
            'principal_point_v': principal_v,
            'skew': skew,
            'aspect_ratio': aspect_ratio,
        }

    except Exception as e:
        print(f"Warning: Failed to parse {xmp_path}: {e}")
        return None


def main():
    """Create comprehensive analysis CSV."""
    print("="*80)
    print("CREATING COMPREHENSIVE ANALYSIS CSV")
    print("="*80)

    root = Path(__file__).parent.parent

    # Load predictions and ground truth
    print("\nLoading data...")
    pred_dict = load_csv(Path(__file__).parent / "train_predictions.csv")
    gt_dict = load_csv(root / "data" / "train_data" / "train_pos.csv")

    print(f"  Predictions: {len(pred_dict)} images")
    print(f"  Ground truth: {len(gt_dict)} images")

    # Find images with both prediction and ground truth
    common_ids = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
    print(f"  Images with both: {len(common_ids)}")

    # Collect all data
    rows = []
    n_missing_xmp = 0

    print("\nProcessing images...")
    for img_id in common_ids:
        pred_x, pred_y = pred_dict[img_id]
        gt_x, gt_y = gt_dict[img_id]

        # Load camera parameters
        xmp_path = root / "rough_matching" / "realityscan_positions" / "train" / f"{img_id:04d}.xmp"
        camera_params = read_camera_params_from_xmp(xmp_path)

        if camera_params is None:
            n_missing_xmp += 1
            # Still include the image but with null camera params
            row = {
                'id': img_id,
                'predicted_x': pred_x,
                'predicted_y': pred_y,
                'gt_x': gt_x,
                'gt_y': gt_y,
                'error_px': math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2),
                'roll_deg': None,
                'pitch_deg': None,
                'yaw_deg': None,
                'focal_length_35mm': None,
                'principal_point_u': None,
                'principal_point_v': None,
                'skew': None,
                'aspect_ratio': None,
            }
            # Add rotation matrix elements (R11-R33)
            for i in range(9):
                row[f'R{i//3+1}{i%3+1}'] = None
            # Add distortion coefficients
            for i in range(6):
                row[f'k{i+1}'] = None
        else:
            row = {
                'id': img_id,
                'predicted_x': pred_x,
                'predicted_y': pred_y,
                'gt_x': gt_x,
                'gt_y': gt_y,
                'error_px': math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2),
                'roll_deg': camera_params['roll_deg'],
                'pitch_deg': camera_params['pitch_deg'],
                'yaw_deg': camera_params['yaw_deg'],
                'focal_length_35mm': camera_params['focal_length_35mm'],
                'principal_point_u': camera_params['principal_point_u'],
                'principal_point_v': camera_params['principal_point_v'],
                'skew': camera_params['skew'],
                'aspect_ratio': camera_params['aspect_ratio'],
            }

            # Add rotation matrix elements (R11, R12, R13, R21, R22, R23, R31, R32, R33)
            R = camera_params['rotation_matrix']
            for i in range(3):
                for j in range(3):
                    row[f'R{i+1}{j+1}'] = R[i, j]

            # Add distortion coefficients (k1-k6)
            for i, k in enumerate(camera_params['distortion_coeffs']):
                row[f'k{i+1}'] = k

        rows.append(row)

    # Define column order
    fieldnames = [
        'id',
        'predicted_x', 'predicted_y',
        'gt_x', 'gt_y',
        'error_px',
        'roll_deg', 'pitch_deg', 'yaw_deg',
        'R11', 'R12', 'R13',
        'R21', 'R22', 'R23',
        'R31', 'R32', 'R33',
        'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
        'focal_length_35mm',
        'principal_point_u', 'principal_point_v',
        'skew', 'aspect_ratio',
    ]

    # Write CSV
    output_path = Path(__file__).parent / "train_analysis.csv"
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Wrote analysis CSV to: {output_path}")
    print(f"  Total images: {len(rows)}")
    print(f"  Missing XMP files: {n_missing_xmp}")

    # Print statistics
    errors = [row['error_px'] for row in rows]
    print(f"\nError statistics:")
    print(f"  Mean:   {np.mean(errors):.1f}px")
    print(f"  Median: {np.median(errors):.1f}px")
    print(f"  Min:    {np.min(errors):.1f}px")
    print(f"  Max:    {np.max(errors):.1f}px")

    print("\nColumns in CSV:")
    print("  - id, predicted_x, predicted_y, gt_x, gt_y, error_px")
    print("  - roll_deg, pitch_deg, yaw_deg (Euler angles)")
    print("  - R11-R33 (rotation matrix elements)")
    print("  - k1-k6 (distortion coefficients)")
    print("  - focal_length_35mm, principal_point_u, principal_point_v")
    print("  - skew, aspect_ratio")

    print("="*80)


if __name__ == "__main__":
    main()
