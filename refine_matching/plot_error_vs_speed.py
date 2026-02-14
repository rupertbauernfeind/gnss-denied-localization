#!/usr/bin/env python3
"""
Plot localization error as a function of motion speed.

Analyzes correlation between drone motion speed (computed from consecutive
rough positions) and localization errors in both x and y directions.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
            # Skip rows with missing data
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
    """
    Compute motion speeds from consecutive rough positions and extract error components.

    Returns:
        dict with arrays: speeds_x, speeds_y, speeds_mag, errors_x, errors_y,
                         errors_abs_x, errors_abs_y, errors_px, ids
    """
    # Create lookup dict for analysis data
    analysis_lookup = {}
    for i, img_id in enumerate(analysis_data['id']):
        analysis_lookup[img_id] = {
            'predicted_x': analysis_data['predicted_x'][i],
            'predicted_y': analysis_data['predicted_y'][i],
            'gt_x': analysis_data['gt_x'][i],
            'gt_y': analysis_data['gt_y'][i],
            'error_px': analysis_data['error_px'][i],
        }

    # Sort IDs for chronological processing
    sorted_ids = sorted(rough_positions.keys())

    # Storage for results
    speeds_x = []
    speeds_y = []
    speeds_mag = []
    errors_x = []
    errors_y = []
    errors_abs_x = []
    errors_abs_y = []
    errors_px = []
    ids = []

    # Process consecutive pairs
    for i in range(len(sorted_ids) - 1):
        id_curr = sorted_ids[i]
        id_next = sorted_ids[i + 1]

        # Only process consecutive IDs
        if id_next - id_curr != 1:
            continue

        # Check both IDs have analysis data
        if id_next not in analysis_lookup or id_curr not in rough_positions:
            continue

        # Compute motion from rough positions
        x_curr, y_curr = rough_positions[id_curr]
        x_next, y_next = rough_positions[id_next]

        dx = x_next - x_curr
        dy = y_next - y_curr
        speed_mag = np.sqrt(dx**2 + dy**2)

        # Get error components for the next position
        analysis = analysis_lookup[id_next]
        error_x = analysis['predicted_x'] - analysis['gt_x']
        error_y = analysis['predicted_y'] - analysis['gt_y']

        # Store results
        speeds_x.append(dx)
        speeds_y.append(dy)
        speeds_mag.append(speed_mag)
        errors_x.append(error_x)
        errors_y.append(error_y)
        errors_abs_x.append(abs(error_x))
        errors_abs_y.append(abs(error_y))
        errors_px.append(analysis['error_px'])
        ids.append(id_next)

    return {
        'speeds_x': np.array(speeds_x),
        'speeds_y': np.array(speeds_y),
        'speeds_mag': np.array(speeds_mag),
        'errors_x': np.array(errors_x),
        'errors_y': np.array(errors_y),
        'errors_abs_x': np.array(errors_abs_x),
        'errors_abs_y': np.array(errors_abs_y),
        'errors_px': np.array(errors_px),
        'ids': np.array(ids),
    }


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("PLOTTING LOCALIZATION ERROR VS MOTION SPEED")
    print("="*80)

    # Define paths
    script_dir = Path(__file__).parent
    rough_csv_path = script_dir / "train_predictions.csv"
    analysis_csv_path = script_dir / "train_analysis.csv"

    # Check if files exist
    if not rough_csv_path.exists():
        print(f"\n✗ Rough positions CSV not found at {rough_csv_path}")
        return

    if not analysis_csv_path.exists():
        print(f"\n✗ Analysis CSV not found at {analysis_csv_path}")
        print("Run create_analysis_csv.py first to generate the data.")
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

    if len(data['speeds_mag']) == 0:
        print("✗ No consecutive pairs found")
        return

    print(f"  Consecutive pairs found: {len(data['speeds_mag'])} pairs")

    # Filter outliers - both speed and error
    MAX_ERROR = 150.0
    MAX_SPEED = 500.0  # px/frame in each dimension

    speeds_x_all = data['speeds_x']
    speeds_y_all = data['speeds_y']
    speeds_mag_all = data['speeds_mag']
    errors_x_all = data['errors_x']
    errors_y_all = data['errors_y']
    errors_abs_x_all = data['errors_abs_x']
    errors_abs_y_all = data['errors_abs_y']
    errors_px_all = data['errors_px']

    # Combined filter: reasonable speeds AND reasonable errors
    speed_mask = (np.abs(speeds_x_all) <= MAX_SPEED) & (np.abs(speeds_y_all) <= MAX_SPEED)
    error_mask = errors_px_all <= MAX_ERROR
    mask = speed_mask & error_mask

    speeds_x = speeds_x_all[mask]
    speeds_y = speeds_y_all[mask]
    speeds_mag = speeds_mag_all[mask]
    errors_x = errors_x_all[mask]
    errors_y = errors_y_all[mask]
    errors_abs_x = errors_abs_x_all[mask]
    errors_abs_y = errors_abs_y_all[mask]
    errors_px = errors_px_all[mask]

    n_filtered_speed = (~speed_mask).sum()
    n_filtered_error = (~error_mask).sum()
    n_filtered_total = len(speeds_mag_all) - len(speeds_mag)

    print(f"\nFiltering outliers:")
    print(f"  Total pairs: {len(speeds_mag_all)}")
    print(f"  Filtered (speed > {MAX_SPEED}px/frame): {n_filtered_speed}")
    print(f"  Filtered (error > {MAX_ERROR}px): {n_filtered_error}")
    print(f"  Filtered (total): {n_filtered_total}")
    print(f"  Remaining: {len(speeds_mag)}")

    # Print statistics
    print(f"\nSpeed statistics (filtered, px/frame):")
    print(f"  X-speed range: {speeds_x.min():.1f} to {speeds_x.max():.1f}")
    print(f"  Y-speed range: {speeds_y.min():.1f} to {speeds_y.max():.1f}")
    print(f"  Speed magnitude range: {speeds_mag.min():.1f} to {speeds_mag.max():.1f}")
    print(f"  Mean speed magnitude: {speeds_mag.mean():.1f}")

    print(f"\nError statistics (filtered):")
    print(f"  X-error range: {errors_x.min():.1f}px to {errors_x.max():.1f}px")
    print(f"  Y-error range: {errors_y.min():.1f}px to {errors_y.max():.1f}px")
    print(f"  Mean |x-error|: {errors_abs_x.mean():.1f}px")
    print(f"  Mean |y-error|: {errors_abs_y.mean():.1f}px")
    print(f"  Mean total error: {errors_px.mean():.1f}px")
    print(f"  RMSE: {np.sqrt((errors_px**2).mean()):.1f}px")

    # Compute error rates (deviation per unit speed)
    # Avoid division by very small speeds
    MIN_SPEED = 10.0  # px/frame minimum for rate calculation
    speed_mask_for_rate = speeds_mag >= MIN_SPEED

    error_rate_total = errors_px / speeds_mag
    error_rate_x = errors_x / speeds_mag  # signed
    error_rate_y = errors_y / speeds_mag  # signed

    # For component-wise rates, use component speeds
    x_speed_nonzero = np.abs(speeds_x) >= MIN_SPEED
    y_speed_nonzero = np.abs(speeds_y) >= MIN_SPEED

    error_rate_x_comp = np.full_like(errors_abs_x, np.nan)
    error_rate_y_comp = np.full_like(errors_abs_y, np.nan)
    error_rate_x_comp[x_speed_nonzero] = errors_abs_x[x_speed_nonzero] / np.abs(speeds_x[x_speed_nonzero])
    error_rate_y_comp[y_speed_nonzero] = errors_abs_y[y_speed_nonzero] / np.abs(speeds_y[y_speed_nonzero])

    print(f"\nError rates (deviation per unit speed):")
    print(f"  Total error rate: {error_rate_total.mean():.3f} (error/speed)")
    print(f"  X error rate: {error_rate_x.mean():.3f} (signed)")
    print(f"  Y error rate: {error_rate_y.mean():.3f} (signed)")
    print(f"  X error rate (component): {np.nanmean(error_rate_x_comp):.3f}")
    print(f"  Y error rate (component): {np.nanmean(error_rate_y_comp):.3f}")

    # Compute correlations - both absolute errors and error rates
    corr_x_signed = np.corrcoef(speeds_x, errors_x)[0, 1]
    corr_y_signed = np.corrcoef(speeds_y, errors_y)[0, 1]
    corr_x_abs = np.corrcoef(np.abs(speeds_x), errors_abs_x)[0, 1]
    corr_y_abs = np.corrcoef(np.abs(speeds_y), errors_abs_y)[0, 1]
    corr_mag = np.corrcoef(speeds_mag, errors_px)[0, 1]

    # Cross-correlations (motion in one direction vs error in perpendicular direction)
    corr_x_to_y_signed = np.corrcoef(speeds_x, errors_y)[0, 1]  # x-speed vs y-error
    corr_y_to_x_signed = np.corrcoef(speeds_y, errors_x)[0, 1]  # y-speed vs x-error
    corr_x_to_y_abs = np.corrcoef(np.abs(speeds_x), errors_abs_y)[0, 1]
    corr_y_to_x_abs = np.corrcoef(np.abs(speeds_y), errors_abs_x)[0, 1]

    # Correlations with error rates (deviation per speed)
    corr_mag_rate = np.corrcoef(speeds_mag, error_rate_total)[0, 1]
    corr_x_rate = np.corrcoef(speeds_x, error_rate_x)[0, 1]
    corr_y_rate = np.corrcoef(speeds_y, error_rate_y)[0, 1]

    # Cross-correlations with error rates
    corr_x_to_y_rate = np.corrcoef(np.abs(speeds_x), errors_abs_y / speeds_mag)[0, 1]
    corr_y_to_x_rate = np.corrcoef(np.abs(speeds_y), errors_abs_x / speeds_mag)[0, 1]

    print(f"\nDirect Correlations - Absolute Errors:")
    print(f"  x_speed vs x_error (signed):  {corr_x_signed:+.3f}")
    print(f"  y_speed vs y_error (signed):  {corr_y_signed:+.3f}")
    print(f"  |x_speed| vs |x_error|:       {corr_x_abs:+.3f}")
    print(f"  |y_speed| vs |y_error|:       {corr_y_abs:+.3f}")
    print(f"  speed_mag vs error_px:        {corr_mag:+.3f}")

    print(f"\nDirect Correlations - Error Rates (deviation/speed):")
    print(f"  x_speed vs x_error_rate:      {corr_x_rate:+.3f}")
    print(f"  y_speed vs y_error_rate:      {corr_y_rate:+.3f}")
    print(f"  speed_mag vs error_rate:      {corr_mag_rate:+.3f}")

    print(f"\nCross-Correlations - Absolute Errors:")
    print(f"  x_speed vs y_error (signed):  {corr_x_to_y_signed:+.3f}")
    print(f"  y_speed vs x_error (signed):  {corr_y_to_x_signed:+.3f}")
    print(f"  |x_speed| vs |y_error|:       {corr_x_to_y_abs:+.3f}")
    print(f"  |y_speed| vs |x_error|:       {corr_y_to_x_abs:+.3f}")

    print(f"\nCross-Correlations - Error Rates:")
    print(f"  |x_speed| vs y_error_rate:    {corr_x_to_y_rate:+.3f}")
    print(f"  |y_speed| vs x_error_rate:    {corr_y_to_x_rate:+.3f}")

    # Create bins for speed magnitude
    bin_width = 25  # px/frame
    speed_bins = np.arange(0, speeds_mag.max() + bin_width, bin_width)
    bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

    # Calculate statistics for each bin
    bin_rmse = []
    bin_mean_errors = []
    bin_mean_error_rates = []
    bin_counts = []

    for i in range(len(speed_bins) - 1):
        bin_mask = (speeds_mag >= speed_bins[i]) & (speeds_mag < speed_bins[i+1])
        bin_errors = errors_px[bin_mask]
        bin_error_rates = error_rate_total[bin_mask]

        if len(bin_errors) > 0:
            bin_rmse.append(np.sqrt(np.mean(bin_errors**2)))
            bin_mean_errors.append(np.mean(bin_errors))
            bin_mean_error_rates.append(np.mean(bin_error_rates))
            bin_counts.append(len(bin_errors))
        else:
            bin_rmse.append(np.nan)
            bin_mean_errors.append(np.nan)
            bin_mean_error_rates.append(np.nan)
            bin_counts.append(0)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Speed magnitude distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(speeds_mag, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(speeds_mag.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {speeds_mag.mean():.1f}px/frame')
    ax.axvline(np.median(speeds_mag), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(speeds_mag):.1f}px/frame')
    ax.set_xlabel('Speed Magnitude (px/frame)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Speed Distribution\n{len(speeds_mag)} consecutive pairs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: x_speed vs x_error_rate (deviation per speed)
    ax = fig.add_subplot(gs[0, 1])
    scatter = ax.scatter(speeds_x, error_rate_x, c=error_rate_total, cmap='RdYlGn_r',
                        s=30, alpha=0.6, vmin=0, vmax=1.0)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('X Speed (px/frame)')
    ax.set_ylabel('X Error Rate (error/speed)')
    ax.set_title(f'X Speed vs X Error Rate\nCorr: {corr_x_rate:+.3f}')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Error Rate')

    # Plot 3: y_speed vs y_error_rate (deviation per speed)
    ax = fig.add_subplot(gs[0, 2])
    scatter = ax.scatter(speeds_y, error_rate_y, c=error_rate_total, cmap='RdYlGn_r',
                        s=30, alpha=0.6, vmin=0, vmax=1.0)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Y Speed (px/frame)')
    ax.set_ylabel('Y Error Rate (error/speed)')
    ax.set_title(f'Y Speed vs Y Error Rate\nCorr: {corr_y_rate:+.3f}')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Error Rate')

    # Plot 4: |x_speed| vs y_error_rate - CROSS-CORRELATION (perpendicular)
    ax = fig.add_subplot(gs[1, 0])
    y_error_rate = errors_abs_y / speeds_mag
    scatter = ax.scatter(np.abs(speeds_x), y_error_rate, c=error_rate_total, cmap='RdYlGn_r',
                        s=30, alpha=0.6, vmin=0, vmax=1.0)
    ax.set_xlabel('|X Speed| (px/frame)')
    ax.set_ylabel('Y Error Rate (y_error/speed)')
    ax.set_title(f'X-Speed vs Y-Error Rate (CROSS)\nCorr: {corr_x_to_y_rate:+.3f}')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Error Rate')

    # Plot 5: |y_speed| vs x_error_rate - CROSS-CORRELATION (perpendicular)
    ax = fig.add_subplot(gs[1, 1])
    x_error_rate = errors_abs_x / speeds_mag
    scatter = ax.scatter(np.abs(speeds_y), x_error_rate, c=error_rate_total, cmap='RdYlGn_r',
                        s=30, alpha=0.6, vmin=0, vmax=1.0)
    ax.set_xlabel('|Y Speed| (px/frame)')
    ax.set_ylabel('X Error Rate (x_error/speed)')
    ax.set_title(f'Y-Speed vs X-Error Rate (CROSS)\nCorr: {corr_y_to_x_rate:+.3f}')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Error Rate')

    # Plot 6: Speed magnitude vs error rate (binned)
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(bin_centers, bin_mean_error_rates, width=bin_width*0.8, alpha=0.7,
           color='orange', edgecolor='black')
    ax.set_xlabel('Speed Magnitude (px/frame)')
    ax.set_ylabel('Mean Error Rate (error/speed)')
    ax.set_title(f'Speed vs Error Rate (binned)\nCorr: {corr_mag_rate:+.3f}')
    ax.grid(True, alpha=0.3)

    # Plot 7: Speed vs error rate scatter (magnitude)
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(speeds_mag, error_rate_total, c=error_rate_total, cmap='RdYlGn_r',
              s=20, alpha=0.6, vmin=0, vmax=1.0)
    ax.set_xlabel('Speed Magnitude (px/frame)')
    ax.set_ylabel('Error Rate (error/speed)')
    ax.set_title(f'Speed vs Error Rate\nCorr: {corr_mag_rate:+.3f}')
    ax.grid(True, alpha=0.3)

    # Plot 8: Sample distribution by speed bins
    ax = fig.add_subplot(gs[2, 1])
    ax.bar(bin_centers, bin_counts, width=bin_width*0.8, alpha=0.7,
           color='green', edgecolor='black')
    ax.set_xlabel('Speed Magnitude (px/frame)')
    ax.set_ylabel('Number of Pairs')
    ax.set_title('Sample Distribution by Speed')
    ax.grid(True, alpha=0.3)

    # Plot 9: Correlation summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary_text = f"""Speed vs Error Rate Analysis

ERROR RATES (deviation/speed):
Direct:
  x_speed vs x_err_rate: {corr_x_rate:+.3f}
  y_speed vs y_err_rate: {corr_y_rate:+.3f}
  speed_mag vs err_rate: {corr_mag_rate:+.3f}

Cross (perpendicular):
  x_speed vs y_err_rate: {corr_x_to_y_rate:+.3f}
  y_speed vs x_err_rate: {corr_y_to_x_rate:+.3f}

ABSOLUTE ERRORS (cross):
  |x_speed| vs |y_error|: {corr_x_to_y_abs:+.3f}
  |y_speed| vs |x_error|: {corr_y_to_x_abs:+.3f}

Stats:
Speed: {speeds_mag.mean():.1f} px/frame
Error rate: {error_rate_total.mean():.3f}
RMSE: {np.sqrt((errors_px**2).mean()):.1f}px
Filtered: {len(speeds_mag)}/{len(speeds_mag_all)}
"""
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    # Save figure
    output_path = script_dir / "error_vs_speed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    # Print binned statistics
    print(f"\nStatistics by Speed Range (errors ≤{MAX_ERROR}px):")
    print(f"{'Speed Range':<25} {'Count':<8} {'RMSE':<12} {'Mean Error':<12} {'Error Rate':<12}")
    print("-"*80)
    for i, center in enumerate(bin_centers):
        if bin_counts[i] > 0:
            speed_start = speed_bins[i]
            speed_end = speed_bins[i+1]
            print(f"{speed_start:>6.1f} to {speed_end:<6.1f} px/frame {bin_counts[i]:<8} "
                  f"{bin_rmse[i]:<12.1f} {bin_mean_errors[i]:<12.1f} {bin_mean_error_rates[i]:<12.3f}")

    plt.show()

    print("="*80)


if __name__ == "__main__":
    main()
