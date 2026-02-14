#!/usr/bin/env python3
"""
Plot MSE coordinate error as a function of camera yaw angle.

Shows if prediction errors correlate with camera orientation.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_analysis_csv(path):
    """Load the analysis CSV file."""
    data = {
        'id': [],
        'error_px': [],
        'yaw_deg': [],
        'pitch_deg': [],
        'roll_deg': [],
    }

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing data
            if row['error_px'] == '' or row['yaw_deg'] == '' or row['yaw_deg'] == 'None':
                continue

            data['id'].append(int(row['id']))
            data['error_px'].append(float(row['error_px']))
            data['yaw_deg'].append(float(row['yaw_deg']))
            data['pitch_deg'].append(float(row['pitch_deg']))
            data['roll_deg'].append(float(row['roll_deg']))

    return data


def main():
    """Plot MSE error vs yaw angle."""
    print("="*80)
    print("PLOTTING MSE ERROR VS YAW ANGLE")
    print("="*80)

    # Check if analysis CSV exists
    csv_path = Path(__file__).parent / "train_analysis.csv"

    if not csv_path.exists():
        print(f"\n✗ Analysis CSV not found at {csv_path}")
        print("Run create_analysis_csv.py first to generate the data.")
        return

    # Load data
    print("\nLoading data...")
    data = load_analysis_csv(csv_path)

    if len(data['error_px']) == 0:
        print("✗ No valid data found in CSV")
        return

    print(f"  Loaded {len(data['error_px'])} images with valid camera data")

    errors_all = np.array(data['error_px'])
    yaws_all = np.array(data['yaw_deg'])
    pitches_all = np.array(data['pitch_deg'])
    rolls_all = np.array(data['roll_deg'])

    # Filter errors <= 150px to remove outliers
    MAX_ERROR = 150.0
    mask = errors_all <= MAX_ERROR
    errors = errors_all[mask]
    yaws = yaws_all[mask]
    pitches = pitches_all[mask]
    rolls = rolls_all[mask]

    n_filtered = len(errors_all) - len(errors)

    print(f"\nFiltering outliers:")
    print(f"  Total images: {len(errors_all)}")
    print(f"  Filtered (error > {MAX_ERROR}px): {n_filtered}")
    print(f"  Remaining: {len(errors)}")

    # Calculate squared errors for MSE
    squared_errors = errors ** 2

    print(f"\nFiltered data (errors ≤ {MAX_ERROR}px):")
    print(f"  Yaw range: {yaws.min():.1f}° to {yaws.max():.1f}°")
    print(f"  Error range: {errors.min():.1f}px to {errors.max():.1f}px")
    print(f"  Mean error: {errors.mean():.1f}px")
    print(f"  MSE: {squared_errors.mean():.1f}px²")
    print(f"  RMSE: {np.sqrt(squared_errors.mean()):.1f}px")

    # Create bins for yaw angles (e.g., 10-degree bins)
    bin_width = 10  # degrees
    yaw_bins = np.arange(yaws.min() - bin_width/2, yaws.max() + bin_width, bin_width)
    bin_centers = (yaw_bins[:-1] + yaw_bins[1:]) / 2

    # Calculate MSE and RMSE for each bin
    bin_mse = []
    bin_rmse = []
    bin_counts = []
    bin_mean_errors = []

    for i in range(len(yaw_bins) - 1):
        bin_mask = (yaws >= yaw_bins[i]) & (yaws < yaw_bins[i+1])
        bin_errors = errors[bin_mask]

        if len(bin_errors) > 0:
            bin_squared_errors = bin_errors ** 2
            mse = np.mean(bin_squared_errors)
            bin_mse.append(mse)
            bin_rmse.append(np.sqrt(mse))
            bin_mean_errors.append(np.mean(bin_errors))
            bin_counts.append(len(bin_errors))
        else:
            bin_mse.append(np.nan)
            bin_rmse.append(np.nan)
            bin_mean_errors.append(np.nan)
            bin_counts.append(0)

    # Create comprehensive figure with camera parameter analysis
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: RMSE Histogram (frequency distribution)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.sqrt(squared_errors.mean()), color='red', linestyle='--', linewidth=2, label=f'RMSE: {np.sqrt(squared_errors.mean()):.1f}px')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f}px')
    ax.set_xlabel('Error (px)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'RMSE Distribution (errors ≤{MAX_ERROR}px)\n{len(errors)} images')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: RMSE vs Yaw (binned)
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(bin_centers, bin_rmse, width=bin_width*0.8, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('RMSE (px)')
    ax.set_title(f'RMSE vs Yaw ({bin_width}° bins)')
    ax.grid(True, alpha=0.3)

    # Plot 3: RMSE vs Pitch (binned)
    ax = fig.add_subplot(gs[0, 2])
    pitch_bins = np.arange(pitches.min() - 5, pitches.max() + 5, 10)
    pitch_centers = (pitch_bins[:-1] + pitch_bins[1:]) / 2
    pitch_rmse = []
    for i in range(len(pitch_bins) - 1):
        mask_pitch = (pitches >= pitch_bins[i]) & (pitches < pitch_bins[i+1])
        if mask_pitch.sum() > 0:
            pitch_rmse.append(np.sqrt(np.mean(errors[mask_pitch]**2)))
        else:
            pitch_rmse.append(np.nan)
    ax.bar(pitch_centers, pitch_rmse, width=8, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Pitch (degrees)')
    ax.set_ylabel('RMSE (px)')
    ax.set_title('RMSE vs Pitch (10° bins)')
    ax.grid(True, alpha=0.3)

    # Plot 4: RMSE vs Roll (binned)
    ax = fig.add_subplot(gs[1, 0])
    roll_bins = np.arange(rolls.min() - 5, rolls.max() + 5, 10)
    roll_centers = (roll_bins[:-1] + roll_bins[1:]) / 2
    roll_rmse = []
    for i in range(len(roll_bins) - 1):
        mask_roll = (rolls >= roll_bins[i]) & (rolls < roll_bins[i+1])
        if mask_roll.sum() > 0:
            roll_rmse.append(np.sqrt(np.mean(errors[mask_roll]**2)))
        else:
            roll_rmse.append(np.nan)
    ax.bar(roll_centers, roll_rmse, width=8, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Roll (degrees)')
    ax.set_ylabel('RMSE (px)')
    ax.set_title('RMSE vs Roll (10° bins)')
    ax.grid(True, alpha=0.3)

    # Plot 5: Error scatter - Yaw vs Pitch (colored by error)
    ax = fig.add_subplot(gs[1, 1])
    scatter = ax.scatter(yaws, pitches, c=errors, cmap='RdYlGn_r', s=30, alpha=0.6,
                        vmin=0, vmax=MAX_ERROR)
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_title('Yaw vs Pitch (colored by error)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error (px)')

    # Plot 6: Error scatter - Yaw vs Roll (colored by error)
    ax = fig.add_subplot(gs[1, 2])
    scatter = ax.scatter(yaws, rolls, c=errors, cmap='RdYlGn_r', s=30, alpha=0.6,
                        vmin=0, vmax=MAX_ERROR)
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Roll (degrees)')
    ax.set_title('Yaw vs Roll (colored by error)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error (px)')

    # Plot 7: Sample distribution by yaw
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(bin_centers, bin_counts, width=bin_width*0.8, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Number of Images')
    ax.set_title(f'Sample Distribution by Yaw')
    ax.grid(True, alpha=0.3)

    # Plot 8: Error vs Yaw scatter
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(yaws, errors, c=errors, cmap='RdYlGn_r', s=20, alpha=0.6,
              vmin=0, vmax=MAX_ERROR)
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Error (px)')
    ax.set_title('Error vs Yaw (scatter)')
    ax.grid(True, alpha=0.3)

    # Plot 9: Correlations summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    corr_yaw = np.corrcoef(yaws, errors)[0, 1]
    corr_pitch = np.corrcoef(pitches, errors)[0, 1]
    corr_roll = np.corrcoef(rolls, errors)[0, 1]

    summary_text = f"""Camera Parameter Correlations

Yaw vs Error:   {corr_yaw:+.3f}
Pitch vs Error: {corr_pitch:+.3f}
Roll vs Error:  {corr_roll:+.3f}

Overall Statistics:
RMSE: {np.sqrt(squared_errors.mean()):.1f}px
Mean: {errors.mean():.1f}px
Median: {np.median(errors):.1f}px
Std: {errors.std():.1f}px

Filtered: {len(errors)}/{len(errors_all)} images
Max error shown: {MAX_ERROR}px
"""
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    # Save figure
    output_path = Path(__file__).parent / "error_vs_yaw.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    # Print statistics by yaw range
    print(f"\nStatistics by Yaw Range (errors ≤{MAX_ERROR}px):")
    print(f"{'Yaw Range':<20} {'Count':<8} {'MSE':<12} {'RMSE':<12} {'Mean Error':<12}")
    print("-"*72)
    for i, center in enumerate(bin_centers):
        if bin_counts[i] > 0:
            yaw_start = yaw_bins[i]
            yaw_end = yaw_bins[i+1]
            print(f"{yaw_start:>6.1f}° to {yaw_end:<6.1f}° {bin_counts[i]:<8} "
                  f"{bin_mse[i]:<12.1f} {bin_rmse[i]:<12.1f} "
                  f"{bin_mean_errors[i]:<12.1f}")

    # Calculate correlation
    correlation = np.corrcoef(yaws, errors)[0, 1]
    print(f"\nCorrelation between yaw and error: {correlation:.3f}")

    plt.show()

    print("="*80)


if __name__ == "__main__":
    main()
