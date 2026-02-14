#!/usr/bin/env python3
"""
Test if a linear model can predict and remove position error based on camera parameters.

Fits: error_px = β₀ + β₁*yaw + β₂*pitch + β₃*roll + β₄*focal_length + ...

Then evaluates if subtracting predicted error improves localization accuracy.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_analysis_csv(path):
    """Load the analysis CSV file."""
    data = {
        'id': [],
        'error_px': [],
        'yaw_deg': [],
        'pitch_deg': [],
        'roll_deg': [],
        'focal_length_35mm': [],
        'principal_point_u': [],
        'principal_point_v': [],
        'skew': [],
        'aspect_ratio': [],
        'k1': [], 'k2': [], 'k3': [], 'k4': [], 'k5': [], 'k6': [],
        'predicted_x': [],
        'predicted_y': [],
        'gt_x': [],
        'gt_y': [],
    }

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing data
            if (row['error_px'] == '' or row['yaw_deg'] == '' or
                row['yaw_deg'] == 'None' or row['pitch_deg'] == 'None'):
                continue

            data['id'].append(int(row['id']))
            data['error_px'].append(float(row['error_px']))
            data['yaw_deg'].append(float(row['yaw_deg']))
            data['pitch_deg'].append(float(row['pitch_deg']))
            data['roll_deg'].append(float(row['roll_deg']))
            data['predicted_x'].append(float(row['predicted_x']))
            data['predicted_y'].append(float(row['predicted_y']))
            data['gt_x'].append(float(row['gt_x']))
            data['gt_y'].append(float(row['gt_y']))

            # Handle potentially missing camera intrinsics
            try:
                data['focal_length_35mm'].append(float(row['focal_length_35mm']))
                data['principal_point_u'].append(float(row['principal_point_u']))
                data['principal_point_v'].append(float(row['principal_point_v']))
                data['skew'].append(float(row['skew']))
                data['aspect_ratio'].append(float(row['aspect_ratio']))
                data['k1'].append(float(row['k1']))
                data['k2'].append(float(row['k2']))
                data['k3'].append(float(row['k3']))
                data['k4'].append(float(row['k4']))
                data['k5'].append(float(row['k5']))
                data['k6'].append(float(row['k6']))
            except (ValueError, KeyError):
                # Remove this entry if intrinsics are missing
                for key in data:
                    data[key].pop()
                continue

    return data


def main():
    """Test linear error correction model."""
    print("="*80)
    print("LINEAR ERROR CORRECTION FROM CAMERA PARAMETERS")
    print("="*80)

    # Load data
    csv_path = Path(__file__).parent / "train_analysis.csv"

    if not csv_path.exists():
        print(f"\n✗ Analysis CSV not found at {csv_path}")
        print("Run create_analysis_csv.py first to generate the data.")
        return

    print("\nLoading data...")
    data = load_analysis_csv(csv_path)

    if len(data['error_px']) == 0:
        print("✗ No valid data found in CSV")
        return

    print(f"  Loaded {len(data['error_px'])} images with complete camera data")

    # Convert to numpy arrays
    errors = np.array(data['error_px'])
    ids = np.array(data['id'])
    pred_x = np.array(data['predicted_x'])
    pred_y = np.array(data['predicted_y'])
    gt_x = np.array(data['gt_x'])
    gt_y = np.array(data['gt_y'])

    # Filter outliers (errors > 150px)
    MAX_ERROR = 150.0
    mask = errors <= MAX_ERROR

    errors_filtered = errors[mask]
    ids_filtered = ids[mask]
    pred_x_filtered = pred_x[mask]
    pred_y_filtered = pred_y[mask]
    gt_x_filtered = gt_x[mask]
    gt_y_filtered = gt_y[mask]

    print(f"\nFiltered to {len(errors_filtered)}/{len(errors)} images (≤{MAX_ERROR}px)")

    # Build feature matrix X (camera parameters)
    feature_names = [
        'yaw_deg', 'pitch_deg', 'roll_deg',
        'focal_length_35mm', 'principal_point_u', 'principal_point_v',
        'skew', 'aspect_ratio',
        'k1', 'k2', 'k3', 'k4', 'k5', 'k6'
    ]

    X_all = []
    for fname in feature_names:
        X_all.append(np.array(data[fname]))
    X_all = np.column_stack(X_all)

    # Filter X to match error filtering
    X = X_all[mask]

    # Target: error_px
    y = errors_filtered

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {', '.join(feature_names)}")

    # Split into train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} images")
    print(f"Test set:  {len(X_test)} images")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit linear regression
    print("\n" + "-"*80)
    print("FITTING LINEAR MODEL: error_px = β₀ + Σ(βᵢ * camera_paramᵢ)")
    print("-"*80)

    model = Ridge(alpha=1.0)  # Using Ridge to prevent overfitting
    model.fit(X_train_scaled, y_train)

    # Predict errors
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate R² score
    train_r2 = model.score(X_train_scaled, y_train)
    test_r2 = model.score(X_test_scaled, y_test)

    print(f"\nModel R² (variance explained):")
    print(f"  Train: {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"  Test:  {test_r2:.4f} ({test_r2*100:.2f}%)")

    # Show feature importance (coefficients)
    print(f"\nFeature Coefficients (top 10 by absolute value):")
    coef_abs = np.abs(model.coef_)
    sorted_idx = np.argsort(coef_abs)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {feature_names[i]:20s}: {model.coef_[i]:+8.4f}")

    # Calculate residual errors (what's left after linear correction)
    residual_train = y_train - y_train_pred
    residual_test = y_test - y_test_pred

    # Statistics
    print("\n" + "="*80)
    print("ERROR STATISTICS - BEFORE AND AFTER LINEAR CORRECTION")
    print("="*80)

    print(f"\n{'Metric':<30} {'Before (Train)':<20} {'After (Train)':<20}")
    print("-"*70)
    print(f"{'Mean error (px)':<30} {y_train.mean():>18.1f}  {np.abs(residual_train).mean():>18.1f}")
    print(f"{'Median error (px)':<30} {np.median(y_train):>18.1f}  {np.median(np.abs(residual_train)):>18.1f}")
    print(f"{'RMSE (px)':<30} {np.sqrt((y_train**2).mean()):>18.1f}  {np.sqrt((residual_train**2).mean()):>18.1f}")
    print(f"{'Std dev (px)':<30} {y_train.std():>18.1f}  {residual_train.std():>18.1f}")

    print(f"\n{'Metric':<30} {'Before (Test)':<20} {'After (Test)':<20}")
    print("-"*70)
    print(f"{'Mean error (px)':<30} {y_test.mean():>18.1f}  {np.abs(residual_test).mean():>18.1f}")
    print(f"{'Median error (px)':<30} {np.median(y_test):>18.1f}  {np.median(np.abs(residual_test)):>18.1f}")
    print(f"{'RMSE (px)':<30} {np.sqrt((y_test**2).mean()):>18.1f}  {np.sqrt((residual_test**2).mean()):>18.1f}")
    print(f"{'Std dev (px)':<30} {y_test.std():>18.1f}  {residual_test.std():>18.1f}")

    # Improvement percentages
    train_mean_improvement = (y_train.mean() - np.abs(residual_train).mean()) / y_train.mean() * 100
    test_mean_improvement = (y_test.mean() - np.abs(residual_test).mean()) / y_test.mean() * 100
    train_rmse_improvement = (np.sqrt((y_train**2).mean()) - np.sqrt((residual_train**2).mean())) / np.sqrt((y_train**2).mean()) * 100
    test_rmse_improvement = (np.sqrt((y_test**2).mean()) - np.sqrt((residual_test**2).mean())) / np.sqrt((y_test**2).mean()) * 100

    print("\n" + "="*80)
    print("IMPROVEMENT FROM LINEAR CORRECTION")
    print("="*80)
    print(f"\nTrain Set:")
    print(f"  Mean error reduction:  {train_mean_improvement:+.1f}%")
    print(f"  RMSE reduction:        {train_rmse_improvement:+.1f}%")

    print(f"\nTest Set:")
    print(f"  Mean error reduction:  {test_mean_improvement:+.1f}%")
    print(f"  RMSE reduction:        {test_rmse_improvement:+.1f}%")

    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Predicted vs actual error (train)
    ax = axes[0, 0]
    ax.scatter(y_train, y_train_pred, alpha=0.5, s=10)
    ax.plot([0, y_train.max()], [0, y_train.max()], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Error (px)')
    ax.set_ylabel('Predicted Error (px)')
    ax.set_title(f'Train: Predicted vs Actual Error\nR²={train_r2:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Predicted vs actual error (test)
    ax = axes[0, 1]
    ax.scatter(y_test, y_test_pred, alpha=0.5, s=10)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Error (px)')
    ax.set_ylabel('Predicted Error (px)')
    ax.set_title(f'Test: Predicted vs Actual Error\nR²={test_r2:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Residual histogram (test)
    ax = axes[0, 2]
    ax.hist(residual_test, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual Error (px)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Test: Residual Distribution\nMean={residual_test.mean():.1f}px')
    ax.grid(True, alpha=0.3)

    # Plot 4: Before histogram (test)
    ax = axes[1, 0]
    ax.hist(y_test, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(y_test.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {y_test.mean():.1f}px')
    ax.set_xlabel('Error (px)')
    ax.set_ylabel('Frequency')
    ax.set_title('Test: Error Distribution BEFORE Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: After histogram (test)
    ax = axes[1, 1]
    ax.hist(np.abs(residual_test), bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.abs(residual_test).mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.abs(residual_test).mean():.1f}px')
    ax.set_xlabel('Absolute Residual (px)')
    ax.set_ylabel('Frequency')
    ax.set_title('Test: Error Distribution AFTER Correction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Feature importance
    ax = axes[1, 2]
    top_n = 10
    top_idx = sorted_idx[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_coefs = [model.coef_[i] for i in top_idx]

    colors = ['green' if c > 0 else 'red' for c in top_coefs]
    ax.barh(range(top_n), top_coefs, color=colors, alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Coefficient')
    ax.set_title('Top 10 Feature Coefficients')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "linear_error_correction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.show()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if test_r2 < 0.05:
        print("\n✗ Linear model explains <5% of error variance")
        print("  Camera parameters have minimal linear relationship with position error")
        print("  Position error is likely dominated by other factors (matching quality, etc.)")
    elif test_r2 < 0.15:
        print(f"\n⚠ Linear model explains {test_r2*100:.1f}% of error variance")
        print("  Some linear relationship exists, but most error comes from other sources")
    else:
        print(f"\n✓ Linear model explains {test_r2*100:.1f}% of error variance")
        print("  Camera parameters have significant linear relationship with position error")
        print("  This correction could be applied to reduce systematic biases")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
