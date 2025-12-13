#!/usr/bin/env python3
"""
Rename calibration images from camera_{camera_number}.jpg_{photo_number}
to camera_{camera_number}_{photo_number}.jpg format.

This script processes files in the 8CM, 12CM, and 16CM subdirectories
and saves converted files to {foldername}_converted directories.
"""
import os
import glob
import re
import shutil

# Root directory containing the calibration image folders
CALIB_IMG_ROOT = os.path.dirname(__file__)
FOLDERS = ['8CM', '12CM', '16CM']

def rename_files_in_folder(folder_path):
    """Copy and rename files from camera_X.jpg_Y to camera_X_Y.jpg in {folder}_converted"""
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    # Create output folder: {folder_name}_converted
    folder_name = os.path.basename(folder_path)
    output_folder = os.path.join(os.path.dirname(folder_path), f"{folder_name}_converted")
    os.makedirs(output_folder, exist_ok=True)
    print(f"  Output folder: {output_folder}")
    
    # Pattern: camera_{number}.jpg_{number}
    pattern = re.compile(r'^camera_(\d+)\.jpg_(\d+)$')
    
    files = os.listdir(folder_path)
    converted_count = 0
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            camera_num = match.group(1)
            photo_num = match.group(2)
            new_name = f"camera_{camera_num}_{photo_num}.jpg"
            
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(output_folder, new_name)
            
            # Check if target already exists
            if os.path.exists(new_path):
                print(f"  Skip (target exists): {filename} -> {new_name}")
                continue
            
            shutil.copy2(old_path, new_path)
            print(f"  Converted: {filename} -> {new_name}")
            converted_count += 1
    
    print(f"  Total converted in {folder_name}: {converted_count}")

if __name__ == "__main__":
    print("Starting calibration image conversion...")
    print(f"Root directory: {CALIB_IMG_ROOT}\n")
    
    for folder_name in FOLDERS:
        folder_path = os.path.join(CALIB_IMG_ROOT, folder_name)
        print(f"Processing folder: {folder_name}")
        rename_files_in_folder(folder_path)
        print()
    
    print("Conversion complete!")
