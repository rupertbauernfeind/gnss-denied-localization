#!/usr/bin/env python3
"""
Predict localization errors from motion speed using linear regression.

This allows you to estimate errors on test data (where GT is unknown) and
correct your predictions to improve accuracy.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_rough_positions(path):
    """Load rough positions from CSV."""
    positions = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = int(row['id'])
            x = float(row['x_pixel'])
            y = float(row['y_pixel'])
            positions[img_id] = (x, y)
    return positions


def load_analysis_csv(path):
    """Load analysis CSV file with error data."""
    data = {
        'id': [],
        'predicted_x': [],
        'predicted_y': [],
        'gt_x': [],
        'gt_y': [],
        'error_px': [],
    }

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['error_px'] == '' or row['gt_x'] == '' or row['gt_y'] == '':
                continue

            data['id'].append(int(row['id']))
            data['predicted_x'].append(float(row['predicted_x']))
            data['predicted_y'].append(float(row['predicted_y']))
            data['gt_x'].append(float(row['gt_x']))
            data['gt_y'].append(float(row['gt_y']))
            data['error_px'].append(float(row['error_px']))

    return data


def compute_speeds_and_errors(rough_positions, analysis_data):
    """Compute motion speeds from consecutive rough positions and extract errors."""
    analysis_lookup = {}
    for i, img_id in enumerate(analysis_data['id']):
        analysis_lookup[img_id] = {
            'predicted_x': analysis_data['predicted_x'][i],
            'predicted_y': analysis_data['predicted_y'][i],
            'gt_x': analysis_data['gt_x'][i],
            'gt_y': analysis_data['gt_y'][i],
            'error_px': analysis_data['error_px'][i],
        }

    sorted_ids = sorted(rough_positions.keys())

    speeds_x = []
    speeds_y = []
    speeds_mag = []
    errors_x = []
    errors_y = []
    errors_px = []
    ids = []

    for i in range(len(sorted_ids) - 1):
        id_curr = sorted_ids[i]
        id_next = sorted_ids[i + 1]

        if id_next - id_curr != 1:
            continue

        if id_next not in analysis_lookup or id_curr not in rough_positions:
            continue

        x_curr, y_curr = rough_positions[id_curr]
        x_next, y_next = rough_positions[id_next]

        dx = x_next - x_curr
        dy = y_next - y_curr
        speed_mag = np.sqrt(dx**2 + dy**2)

        analysis = analysis_lookup[id_next]
        error_x = analysis['predicted_x'] - analysis['gt_x']
        error_y = analysis['predicted_y'] - analysis['gt_y']

        speeds_x.append(dx)
        speeds_y.append(dy)
        speeds_mag.append(speed_mag)
        errors_x.append(error_x)
        errors_y.append(error_y)
        errors_px.append(analysis['error_px'])
        ids.append(id_next)

    return {
        'speeds_x': np.array(speeds_x),
        'speeds_y': np.array(speeds_y),
        'speeds_mag': np.array(speeds_mag),
        'errors_x': np.array(errors_x),
        'errors_y': np.array(errors_y),
        'errors_px': np.array(errors_px),
        'ids': np.array(ids),
    }


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("PREDICTING ERRORS FROM SPEED - LINEAR REGRESSION ANALYSIS")
    print("="*80)

    # Define paths
    script_dir = Path(__file__).parent
    rough_csv_path = script_dir / "train_predictions.csv"
    analysis_csv_path = script_dir / "train_analysis.csv"

    if not rough_csv_path.exists() or not analysis_csv_path.exists():
        print(f"\n✗ Required files not found")
        return

    # Load data
    print("\nLoading data...")
    rough_positions = load_rough_positions(rough_csv_path)
    analysis_data = load_analysis_csv(analysis_csv_path)

    print(f"  Rough positions: {len(rough_positions)} images")
    print(f"  Analysis data: {len(analysis_data['id'])} images")

    # Compute speeds and errors
    print("\nComputing motion speeds...")
    data = compute_speeds_and_errors(rough_positions, analysis_data)

    # Filter outliers
    MAX_ERROR = 150.0
    MAX_SPEED = 500.0

    speeds_x_all = data['speeds_x']
    speeds_y_all = data['speeds_y']
    speeds_mag_all = data['speeds_mag']
    errors_x_all = data['errors_x']
    errors_y_all = data['errors_y']
    errors_px_all = data['errors_px']

    speed_mask = (np.abs(speeds_x_all) <= MAX_SPEED) & (np.abs(speeds_y_all) <= MAX_SPEED)
    error_mask = errors_px_all <= MAX_ERROR
    mask = speed_mask & error_mask

    speeds_x = speeds_x_all[mask]
    speeds_y = speeds_y_all[mask]
    speeds_mag = speeds_mag_all[mask]
    errors_x = errors_x_all[mask]
    errors_y = errors_y_all[mask]
    errors_px = errors_px_all[mask]

    print(f"  Filtered pairs: {len(speeds_x)} (from {len(speeds_x_all)})")

    # Prepare features for regression
    # Simple model: just linear speed components
    X_simple = np.column_stack([speeds_x, speeds_y])

    # Medium model: add quadratic and interaction terms (but not speed_mag to avoid redundancy)
    X_medium = np.column_stack([
        speeds_x,
        speeds_y,
        speeds_x**2,
        speeds_y**2,
        speeds_x * speeds_y,  # interaction term
    ])

    # Full model: also include speed magnitude
    X_full = np.column_stack([
        speeds_x,
        speeds_y,
        speeds_mag,
        speeds_x**2,
        speeds_y**2,
        speeds_x * speeds_y,
    ])

    print("\n" + "="*80)
    print("LINEAR REGRESSION: Predicting X and Y Errors from Speed")
    print("="*80)

    # Model for X error
    print("\n--- Predicting X Error ---")

    # Simple model
    model_x_simple = LinearRegression()
    model_x_simple.fit(X_simple, errors_x)
    pred_x_simple = model_x_simple.predict(X_simple)
    r2_x_simple = r2_score(errors_x, pred_x_simple)
    mse_x_simple = mean_squared_error(errors_x, pred_x_simple)

    # Medium model (with quadratic + interaction)
    model_x_medium = LinearRegression()
    model_x_medium.fit(X_medium, errors_x)
    pred_x_medium = model_x_medium.predict(X_medium)
    r2_x_medium = r2_score(errors_x, pred_x_medium)
    mse_x_medium = mean_squared_error(errors_x, pred_x_medium)

    # Full model
    model_x_full = LinearRegression()
    model_x_full.fit(X_full, errors_x)
    pred_x_full = model_x_full.predict(X_full)
    r2_x_full = r2_score(errors_x, pred_x_full)
    mse_x_full = mean_squared_error(errors_x, pred_x_full)

    print(f"Simple model (speed_x, speed_y):")
    print(f"  R² = {r2_x_simple:.4f}")
    print(f"  MSE = {mse_x_simple:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_x_simple):.2f}px")
    print(f"  Coefficients: speed_x={model_x_simple.coef_[0]:+.4f}, speed_y={model_x_simple.coef_[1]:+.4f}")
    print(f"  Intercept: {model_x_simple.intercept_:+.4f}")

    print(f"\nMedium model (+ quadratic + interaction):")
    print(f"  R² = {r2_x_medium:.4f}")
    print(f"  MSE = {mse_x_medium:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_x_medium):.2f}px")
    print(f"  Features: speed_x, speed_y, speed_x², speed_y², speed_x*speed_y")

    print(f"\nFull model (+ speed_mag):")
    print(f"  R² = {r2_x_full:.4f}")
    print(f"  MSE = {mse_x_full:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_x_full):.2f}px")

    # Model for Y error
    print("\n--- Predicting Y Error ---")

    # Simple model
    model_y_simple = LinearRegression()
    model_y_simple.fit(X_simple, errors_y)
    pred_y_simple = model_y_simple.predict(X_simple)
    r2_y_simple = r2_score(errors_y, pred_y_simple)
    mse_y_simple = mean_squared_error(errors_y, pred_y_simple)

    # Medium model (with quadratic + interaction)
    model_y_medium = LinearRegression()
    model_y_medium.fit(X_medium, errors_y)
    pred_y_medium = model_y_medium.predict(X_medium)
    r2_y_medium = r2_score(errors_y, pred_y_medium)
    mse_y_medium = mean_squared_error(errors_y, pred_y_medium)

    # Full model
    model_y_full = LinearRegression()
    model_y_full.fit(X_full, errors_y)
    pred_y_full = model_y_full.predict(X_full)
    r2_y_full = r2_score(errors_y, pred_y_full)
    mse_y_full = mean_squared_error(errors_y, pred_y_full)

    print(f"Simple model (speed_x, speed_y):")
    print(f"  R² = {r2_y_simple:.4f}")
    print(f"  MSE = {mse_y_simple:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_y_simple):.2f}px")
    print(f"  Coefficients: speed_x={model_y_simple.coef_[0]:+.4f}, speed_y={model_y_simple.coef_[1]:+.4f}")
    print(f"  Intercept: {model_y_simple.intercept_:+.4f}")

    print(f"\nMedium model (+ quadratic + interaction):")
    print(f"  R² = {r2_y_medium:.4f}")
    print(f"  MSE = {mse_y_medium:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_y_medium):.2f}px")
    print(f"  Features: speed_x, speed_y, speed_x², speed_y², speed_x*speed_y")

    print(f"\nFull model (+ speed_mag):")
    print(f"  R² = {r2_y_full:.4f}")
    print(f"  MSE = {mse_y_full:.2f}px²")
    print(f"  RMSE = {np.sqrt(mse_y_full):.2f}px")

    # Compute corrected predictions
    print("\n" + "="*80)
    print("ERROR CORRECTION ANALYSIS")
    print("="*80)

    # Correct the errors using predicted errors from medium model
    corrected_errors_x = errors_x - pred_x_medium
    corrected_errors_y = errors_y - pred_y_medium
    corrected_error_total = np.sqrt(corrected_errors_x**2 + corrected_errors_y**2)

    original_mse = mean_squared_error(np.zeros_like(errors_px), errors_px)
    corrected_mse = mean_squared_error(np.zeros_like(corrected_error_total), corrected_error_total)

    print(f"\nOriginal (uncorrected) predictions:")
    print(f"  RMSE: {np.sqrt(original_mse):.2f}px")
    print(f"  MSE: {original_mse:.2f}px²")
    print(f"  Mean error: {errors_px.mean():.2f}px")

    print(f"\nCorrected predictions (using speed-based error estimates):")
    print(f"  RMSE: {np.sqrt(corrected_mse):.2f}px")
    print(f"  MSE: {corrected_mse:.2f}px²")
    print(f"  Mean error: {corrected_error_total.mean():.2f}px")

    improvement = (original_mse - corrected_mse) / original_mse * 100
    print(f"\nImprovement:")
    print(f"  MSE reduction: {original_mse - corrected_mse:.2f}px² ({improvement:.1f}%)")
    print(f"  RMSE reduction: {np.sqrt(original_mse) - np.sqrt(corrected_mse):.2f}px")

    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: X error - actual vs predicted
    ax = axes[0, 0]
    ax.scatter(errors_x, pred_x_medium, alpha=0.5, s=20)
    lim = max(abs(errors_x.min()), abs(errors_x.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual X Error (px)')
    ax.set_ylabel('Predicted X Error (px)')
    ax.set_title(f'X Error Prediction (Medium Model)\nR²={r2_x_medium:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Y error - actual vs predicted
    ax = axes[0, 1]
    ax.scatter(errors_y, pred_y_medium, alpha=0.5, s=20)
    lim = max(abs(errors_y.min()), abs(errors_y.max()))
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual Y Error (px)')
    ax.set_ylabel('Predicted Y Error (px)')
    ax.set_title(f'Y Error Prediction (Medium Model)\nR²={r2_y_medium:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residuals after correction
    ax = axes[0, 2]
    ax.scatter(speeds_mag, corrected_error_total, alpha=0.5, s=20, c=corrected_error_total,
               cmap='RdYlGn_r', vmin=0, vmax=100)
    ax.set_xlabel('Speed Magnitude (px/frame)')
    ax.set_ylabel('Residual Error After Correction (px)')
    ax.set_title('Residual Errors After Speed-Based Correction')
    ax.grid(True, alpha=0.3)

    # Plot 4: Error distribution before/after
    ax = axes[1, 0]
    ax.hist(errors_px, bins=30, alpha=0.5, label='Original', color='red', edgecolor='black')
    ax.hist(corrected_error_total, bins=30, alpha=0.5, label='Corrected', color='green', edgecolor='black')
    ax.axvline(errors_px.mean(), color='red', linestyle='--', linewidth=2)
    ax.axvline(corrected_error_total.mean(), color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Error Magnitude (px)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution\nRMSE: {np.sqrt(original_mse):.1f}→{np.sqrt(corrected_mse):.1f}px')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: X error residuals
    ax = axes[1, 1]
    ax.scatter(speeds_x, corrected_errors_x, alpha=0.5, s=20)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('X Speed (px/frame)')
    ax.set_ylabel('X Error Residual (px)')
    ax.set_title(f'X Residuals After Correction\nRMSE={np.sqrt(mse_x_medium):.1f}px')
    ax.grid(True, alpha=0.3)

    # Plot 6: Y error residuals
    ax = axes[1, 2]
    ax.scatter(speeds_y, corrected_errors_y, alpha=0.5, s=20)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Y Speed (px/frame)')
    ax.set_ylabel('Y Error Residual (px)')
    ax.set_title(f'Y Residuals After Correction\nRMSE={np.sqrt(mse_y_medium):.1f}px')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = script_dir / "error_prediction_from_speed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    # Save the model coefficients for use on test data
    print("\n" + "="*80)
    print("MODEL COEFFICIENTS (for test data correction)")
    print("="*80)
    print("\nMedium model features: [speed_x, speed_y, speed_x², speed_y², speed_x*speed_y]")
    print("\nTo correct test predictions, apply:")
    print(f"\nX correction:")
    print(f"  error_pred_x = {model_x_medium.coef_[0]:+.6f} * speed_x")
    print(f"               + {model_x_medium.coef_[1]:+.6f} * speed_y")
    print(f"               + {model_x_medium.coef_[2]:+.6f} * speed_x²")
    print(f"               + {model_x_medium.coef_[3]:+.6f} * speed_y²")
    print(f"               + {model_x_medium.coef_[4]:+.6f} * speed_x*speed_y")
    print(f"               + {model_x_medium.intercept_:+.6f}")
    print(f"  corrected_x = predicted_x - error_pred_x")
    print(f"\nY correction:")
    print(f"  error_pred_y = {model_y_medium.coef_[0]:+.6f} * speed_x")
    print(f"               + {model_y_medium.coef_[1]:+.6f} * speed_y")
    print(f"               + {model_y_medium.coef_[2]:+.6f} * speed_x²")
    print(f"               + {model_y_medium.coef_[3]:+.6f} * speed_y²")
    print(f"               + {model_y_medium.coef_[4]:+.6f} * speed_x*speed_y")
    print(f"               + {model_y_medium.intercept_:+.6f}")
    print(f"  corrected_y = predicted_y - error_pred_y")

    plt.show()
    print("="*80)


if __name__ == "__main__":
    main()
