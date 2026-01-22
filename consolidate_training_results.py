"""
Consolidate training results from multiple chunks into a single results file.

Usage:
  python consolidate_training_results.py [--run-dir <path>] [--output <path>]

This script:
  1. Finds the training run directory (optionally specified or auto-detected as latest)
  2. Discovers all results.csv files from individual training chunks
  3. Merges epochs sequentially, preserving chunk boundaries
  4. Saves a consolidated results.csv with all epochs
  5. Includes metadata (run config, chunk info, training duration)
  6. Optionally generates summary plots
"""

import os
import glob
import csv
import json
import argparse
from datetime import datetime
import numpy as np
from pathlib import Path

def find_latest_train_run():
    """Find the most recent training run directory."""
    search_patterns = [
        os.path.join('runs', 'detect', '*'),
        os.path.join('runs', 'train', '*'),
        os.path.join('runs', '*', 'train*'),
        os.path.join('runs', '*', 'exp*'),
    ]
    
    candidates = []
    for pat in search_patterns:
        candidates.extend(glob.glob(pat))
    
    candidates = [p for p in candidates if os.path.isdir(p)]
    
    if not candidates:
        return None
    
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def collect_chunk_results(run_dir):
    """
    Collect all results.csv files from a training run directory and sibling directories.
    
    For chunked training, subprocess runs create separate train* directories.
    This function looks in:
    1. The specified run_dir (freeze stage, often train19)
    2. Sibling train* directories at the same level (chunks from subprocesses)
    3. results_chunk_*.csv files within run_dir
    
    Returns:
      dict: {
        'chunk_files': [results_csv_path, ...],
        'weights_dirs': [weight_dir_paths],
        'total_chunks': int
      }
    """
    chunk_results = []
    weights_dirs = []
    
    # First, look for results_chunk_*.csv files in the main run_dir (new chunk naming convention)
    chunk_csv_files = sorted(glob.glob(os.path.join(run_dir, 'results_chunk_*.csv')))
    chunk_results.extend(chunk_csv_files)
    
    if chunk_csv_files:
        print(f"Found {len(chunk_csv_files)} chunk result files (results_chunk_*.csv) in {run_dir}")
    
    # Also look for results.csv in the main run_dir itself
    main_results_csv = os.path.join(run_dir, 'results.csv')
    if os.path.exists(main_results_csv) and main_results_csv not in chunk_results:
        chunk_results.append(main_results_csv)
    
    # If no chunk files found in run_dir, search sibling train* directories (for backward compatibility)
    if not chunk_csv_files:
        parent_dir = os.path.dirname(run_dir)
        if os.path.isdir(parent_dir):
            # Find all train* directories at the same level
            sibling_trains = sorted(glob.glob(os.path.join(parent_dir, 'train*')))
            for sibling in sibling_trains:
                sibling_results = os.path.join(sibling, 'results.csv')
                if os.path.exists(sibling_results):
                    chunk_results.append(sibling_results)
            
            if sibling_trains:
                print(f"Found {len(sibling_trains)} sibling train* directories, collecting results.csv from each")
    
    # Look for weights directories (track training state)
    for pattern in [os.path.join(run_dir, '*/weights'), os.path.join(run_dir, 'weights')]:
        weights_dirs.extend(glob.glob(pattern))
    
    # Sort by modification time (maintains chunk order)
    chunk_results = sorted(list(set(chunk_results)), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    
    return {
        'chunk_files': chunk_results,
        'weights_dirs': weights_dirs,
        'total_chunks': len(chunk_results)
    }

def read_csv_rows(csv_path):
    """Read CSV and return (header, rows) or (None, None) if failed."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return None, None
            return rows[0], rows[1:]
    except Exception as e:
        print(f"[WARNING] Failed to read {csv_path}: {e}")
        return None, None

def consolidate_results(chunk_files):
    """
    Merge results from multiple chunk CSV files into a single consolidated result.
    
    Each chunk's results.csv has epoch numbers starting from 1. This function:
    1. Reads all chunks
    2. Adjusts epoch column to be sequential across chunks (1-10, 11-20, 21-30, etc.)
    3. Merges into single consolidated result
    
    Returns:
      (header, all_rows, chunk_boundaries)
    """
    all_rows = []
    header = None
    chunk_boundaries = {}  # {chunk_index: (start_epoch, end_epoch)}
    epoch_offset = 0
    epoch_col_idx = None
    
    for chunk_idx, csv_path in enumerate(chunk_files):
        h, rows = read_csv_rows(csv_path)
        
        if h is None or rows is None:
            print(f"[SKIP] {csv_path} (unreadable)")
            continue
        
        if header is None:
            header = h
            # Find epoch column index (first column is typically epoch)
            for i, col_name in enumerate(header):
                if col_name.lower() in ['epoch', ' epoch', 'ep']:
                    epoch_col_idx = i
                    break
            # Fallback: assume first column is epoch if not found
            if epoch_col_idx is None:
                epoch_col_idx = 0
        
        # Adjust epoch numbers in this chunk
        adjusted_rows = []
        for row in rows:
            adjusted_row = list(row)  # Make a copy
            if epoch_col_idx is not None and epoch_col_idx < len(adjusted_row):
                try:
                    original_epoch = int(float(adjusted_row[epoch_col_idx]))
                    new_epoch = epoch_offset + original_epoch
                    adjusted_row[epoch_col_idx] = str(new_epoch)
                except (ValueError, IndexError):
                    pass  # If can't parse epoch, leave it as-is
            adjusted_rows.append(adjusted_row)
        
        # Track where this chunk starts and ends
        start_epoch = epoch_offset + 1
        end_epoch = epoch_offset + len(rows)
        chunk_boundaries[chunk_idx] = (start_epoch, end_epoch)
        
        all_rows.extend(adjusted_rows)
        epoch_offset = end_epoch
        
        print(f"[OK] Chunk {chunk_idx}: {csv_path} ({len(rows)} rows, epochs {start_epoch}-{end_epoch})")
    
    return header, all_rows, chunk_boundaries

def save_consolidated_csv(output_path, header, rows):
    """Save consolidated results to CSV."""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[OK] Saved consolidated results: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {output_path}: {e}")
        return False

def save_consolidation_metadata(metadata_path, run_dir, chunk_info, chunk_boundaries, header, num_rows):
    """Save metadata about the consolidation."""
    metadata = {
        'run_directory': run_dir,
        'consolidation_timestamp': datetime.now().isoformat(),
        'total_chunks': chunk_info['total_chunks'],
        'total_epochs': num_rows,
        'chunk_boundaries': chunk_boundaries,
        'csv_columns': header,
        'weights_directories': chunk_info['weights_dirs'],
    }
    
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[OK] Saved metadata: {metadata_path}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save metadata: {e}")
        return False

def extract_metrics(header, rows, metric_names):
    """Extract specific metrics from rows."""
    indices = {}
    for metric in metric_names:
        try:
            indices[metric] = header.index(metric)
        except ValueError:
            # Try fuzzy matching
            for i, col in enumerate(header):
                if metric.lower() in col.lower():
                    indices[metric] = i
                    break
    
    metrics_data = {}
    for metric, idx in indices.items():
        if idx is not None:
            try:
                values = [float(row[idx]) for row in rows if row and len(row) > idx]
                metrics_data[metric] = {
                    'values': values,
                    'min': min(values) if values else None,
                    'max': max(values) if values else None,
                    'mean': np.mean(values) if values else None,
                    'final': float(rows[-1][idx]) if rows and len(rows[-1]) > idx else None,
                }
            except Exception:
                pass
    
    return metrics_data

def generate_summary_report(run_dir, consolidated_csv, header, rows):
    """Generate a human-readable summary report."""
    report_path = os.path.join(run_dir, 'TRAINING_SUMMARY.txt')
    
    # Extract key metrics
    metric_names = [
        'mAP50', 'mAP_0.5', 'metrics/mAP50',
        'mAP', 'mAP_0.5:0.95', 'metrics/mAP50-95',
        'loss/box_loss', 'loss/cls_loss',
        'val/box_loss', 'val/cls_loss',
    ]
    metrics = extract_metrics(header, rows, metric_names)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING RUN SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Run Directory: {run_dir}\n")
            f.write(f"Consolidated Results: {consolidated_csv}\n")
            f.write(f"Total Epochs: {len(rows)}\n")
            f.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("KEY METRICS\n")
            f.write("=" * 70 + "\n")
            
            for metric_name, metric_data in metrics.items():
                if metric_data['values']:
                    f.write(f"\n{metric_name}:\n")
                    f.write(f"  Initial: {metric_data['values'][0]:.6f}\n")
                    f.write(f"  Final:   {metric_data['final']:.6f}\n")
                    f.write(f"  Min:     {metric_data['min']:.6f}\n")
                    f.write(f"  Max:     {metric_data['max']:.6f}\n")
                    f.write(f"  Mean:    {metric_data['mean']:.6f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("CSV COLUMNS\n")
            f.write("=" * 70 + "\n")
            for i, col in enumerate(header):
                f.write(f"  [{i:2d}] {col}\n")
        
        print(f"[OK] Saved summary report: {report_path}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save summary report: {e}")
        return False

def generate_plots(run_dir, consolidated_csv, header, rows):
    """Generate comparison plots for key metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not available; skipping plot generation")
        return
    
    # Extract metrics for plotting
    plot_metrics = [
        ('mAP50', ['mAP50', 'metrics/mAP50', 'mAP_0.5']),
        ('mAP50-95', ['mAP_0.5:0.95', 'metrics/mAP50-95', 'mAP']),
        ('Box Loss', ['loss/box_loss', 'box_loss']),
        ('Class Loss', ['loss/cls_loss', 'cls_loss']),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax_idx, (plot_name, metric_aliases) in enumerate(plot_metrics):
        if ax_idx >= len(axes):
            break
        
        ax = axes[ax_idx]
        col_idx = None
        
        # Find column index
        for alias in metric_aliases:
            try:
                col_idx = header.index(alias)
                break
            except ValueError:
                for i, col in enumerate(header):
                    if alias.lower() in col.lower():
                        col_idx = i
                        break
        
        if col_idx is not None:
            try:
                values = [float(row[col_idx]) for row in rows if row and len(row) > col_idx]
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, 'b-', linewidth=2, marker='o', markersize=3)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(plot_name)
                ax.set_title(f'{plot_name} Progress')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {plot_name}\n{str(e)}',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'{plot_name} not found in results',
                   ha='center', va='center', transform=ax.transAxes)
    
    plot_path = os.path.join(run_dir, 'consolidated_metrics.png')
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        print(f"[OK] Saved plots: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Failed to save plots: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Consolidate YOLO training results from multiple chunks'
    )
    parser.add_argument(
        '--run-dir',
        default=None,
        help='Training run directory (default: auto-detect latest)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output consolidated CSV path (default: <run-dir>/consolidated_results.csv)'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate comparison plots'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate summary report'
    )
    
    args = parser.parse_args()
    
    # Find run directory
    run_dir = args.run_dir
    if run_dir is None:
        print("[INFO] Auto-detecting latest training run...")
        run_dir = find_latest_train_run()
        if run_dir is None:
            print("[ERROR] No training run directory found. Specify with --run-dir")
            return 1
    
    if not os.path.isdir(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        return 1
    
    print(f"[INFO] Using run directory: {run_dir}")
    
    # Collect chunk results
    print("[INFO] Collecting chunk results...")
    chunk_info = collect_chunk_results(run_dir)
    print(f"[INFO] Found {chunk_info['total_chunks']} chunk(s)")
    
    if not chunk_info['chunk_files']:
        print("[ERROR] No results.csv files found in run directory")
        return 1
    
    # Consolidate
    print("[INFO] Consolidating results...")
    header, all_rows, chunk_boundaries = consolidate_results(chunk_info['chunk_files'])
    
    if header is None or not all_rows:
        print("[ERROR] No data to consolidate")
        return 1
    
    print(f"[INFO] Total epochs consolidated: {len(all_rows)}")
    
    # Save consolidated CSV
    output_csv = args.output
    if output_csv is None:
        output_csv = os.path.join(run_dir, 'consolidated_results.csv')
    
    if not save_consolidated_csv(output_csv, header, all_rows):
        return 1
    
    # Save metadata
    metadata_path = os.path.join(run_dir, 'consolidation_metadata.json')
    save_consolidation_metadata(metadata_path, run_dir, chunk_info, chunk_boundaries, header, len(all_rows))
    
    # Optional: generate report and plots
    if args.report:
        generate_summary_report(run_dir, output_csv, header, all_rows)
    
    if args.plots:
        generate_plots(run_dir, output_csv, header, all_rows)
    
    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"Consolidated Results: {output_csv}")
    print(f"Metadata:             {metadata_path}")
    
    return 0

if __name__ == '__main__':
    exit(main())
