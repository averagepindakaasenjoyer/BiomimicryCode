"""
Server-side training wrapper: runs YoloTrain with automatic result consolidation.

Usage:
  python server_train.py [--consolidate] [--report] [--plots]

This script:
  1. Imports and runs YoloTrain.run_training_validation()
  2. Automatically consolidates results after training completes
  3. Generates summary reports and plots
  4. Logs all output to a timestamped log file
  5. Outputs a single consolidated_results.csv file ready for download
"""

import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Server-side YOLO training with result consolidation')
    parser.add_argument('--consolidate', action='store_true', default=True, help='Consolidate results after training (default: True)')
    parser.add_argument('--report', action='store_true', default=True, help='Generate summary report (default: True)')
    parser.add_argument('--plots', action='store_true', default=False, help='Generate metric plots')
    parser.add_argument('--keep-chunks', action='store_true', default=False, help='Keep individual chunk results (default: False, removes after consolidation)')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SERVER-SIDE YOLO TRAINING")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Consolidate: {args.consolidate}")
    logger.info(f"Report: {args.report}")
    logger.info(f"Plots: {args.plots}")
    logger.info("")
    
    # Import training module
    try:
        from YoloTrain import check_device, run_training_validation
    except ImportError as e:
        logger.error(f"Failed to import YoloTrain: {e}")
        return 1
    
    # Run training
    try:
        logger.info("Starting training...")
        start_time = time.time()
        device = check_device()
        run_training_validation(device)
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/60:.1f} minutes")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    
    # Consolidate results if requested
    if args.consolidate:
        try:
            logger.info("")
            logger.info("Consolidating training results...")
            from consolidate_training_results import find_latest_train_run, collect_chunk_results, consolidate_results, save_consolidated_csv, save_consolidation_metadata, generate_summary_report, generate_plots
            
            run_dir = find_latest_train_run()
            if run_dir is None:
                logger.warning("Could not find training run directory for consolidation")
                return 1
            
            logger.info(f"Run directory: {run_dir}")
            
            chunk_info = collect_chunk_results(run_dir)
            logger.info(f"Found {chunk_info['total_chunks']} chunk(s)")
            
            header, all_rows, chunk_boundaries = consolidate_results(chunk_info['chunk_files'])
            
            if header is None or not all_rows:
                logger.error("No data to consolidate")
                return 1
            
            logger.info(f"Total epochs: {len(all_rows)}")
            
            # Save consolidated CSV
            output_csv = os.path.join(run_dir, 'consolidated_results.csv')
            save_consolidated_csv(output_csv, header, all_rows)
            
            # Save metadata
            metadata_path = os.path.join(run_dir, 'consolidation_metadata.json')
            save_consolidation_metadata(metadata_path, run_dir, chunk_info, chunk_boundaries, header, len(all_rows))
            
            # Generate report
            if args.report:
                logger.info("Generating summary report...")
                generate_summary_report(run_dir, output_csv, header, all_rows)
            
            # Generate plots
            if args.plots:
                logger.info("Generating plots...")
                try:
                    generate_plots(run_dir, output_csv, header, all_rows)
                except Exception as e:
                    logger.warning(f"Plot generation failed: {e}")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Consolidated results: {output_csv}")
            logger.info(f"Metadata: {metadata_path}")
            
            # Optional: clean up individual chunk results
            if not args.keep_chunks:
                logger.info("Removing individual chunk results...")
                try:
                    for csv_file in chunk_info['chunk_files']:
                        if os.path.exists(csv_file) and csv_file != output_csv:
                            os.remove(csv_file)
                            logger.debug(f"Removed: {csv_file}")
                except Exception as e:
                    logger.warning(f"Could not remove all chunk files: {e}")
        
        except Exception as e:
            logger.error(f"Result consolidation failed: {e}", exc_info=True)
            return 1
    
    logger.info(f"Server training complete. Full log available at: {log_file}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
