import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

def find_signal_files(folder):
    """Find all signals and their corresponding .dat, .hea, and -0.png files in the folder."""
    signal_files = []
    
    # Search for all files with pattern -0.png, as each ECG has .dat, .hea, and -0.png
    image_files = sorted([str(p) for p in Path(folder).glob("**/*-0.png")])
    
    print(f"Looking for image files in folder: {folder}")  # Debugging
    print(f"Found {len(image_files)} image files in {folder}")  # Debugging
    
    for image_file in image_files:
        base_name = image_file.replace('-0.png', '')
        dat_file = Path(folder) / f"{base_name}.dat"
        hea_file = Path(folder) / f"{base_name}.hea"
        
        # Check if all required files (.dat, .hea, -0.png) exist for each ECG
        if dat_file.exists() and hea_file.exists():
            signal_files.append((str(image_file), str(dat_file), str(hea_file)))
        else:
            print(f"Warning: Missing .dat or .hea file for {base_name}")  # Debugging
    
    return signal_files

def split_data(signal_files, test_size=0.15, random_state=42):
    """Split the signal files into training and testing sets, keeping .dat, .hea, and .png files together."""
    return train_test_split(signal_files, test_size=test_size, random_state=random_state)

def save_files(files, output_folder, subfolder_name):
    """Save the files in their corresponding folders (train or test)."""
    target_folder = Path(output_folder) / subfolder_name
    target_folder.mkdir(parents=True, exist_ok=True)
    
    for image_file, dat_file, hea_file in files:
        # Copy each set of files (.dat, .hea, .png) to the respective train/test folder
        shutil.copy(image_file, target_folder)
        shutil.copy(dat_file, target_folder)
        shutil.copy(hea_file, target_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split ECG data into train and test sets.")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input data folder")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to save the split data")
    args = parser.parse_args()

    # Find the signal files (.dat, .hea, and -0.png)
    signal_files = find_signal_files(args.input_folder)

    # Check if any files were found
    if len(signal_files) == 0:
        raise ValueError(f"No signal files found in {args.input_folder}. Ensure there are .png, .dat, and .hea files.")
    
    # Split the data into training and testing sets
    train_files, test_files = split_data(signal_files)
    
    # Save the split files into the corresponding train and test folders
    save_files(train_files, args.output_folder, 'train')
    save_files(test_files, args.output_folder, 'test')

    print(f"Data split completed. {len(train_files)} files in train, {len(test_files)} files in test.")
