# data_loader.py

import os
import argparse
import logging
import tensorflow as tf
from pathlib import Path
import wfdb  # Ensure you have wfdb installed: pip install wfdb
import numpy as np
import json

# Define global constants
SIGNAL_INPUT_LENGTH = 5000  # Must match the preprocessing
IMAGE_HEIGHT = 400  # Updated to match model_train.py
IMAGE_WIDTH = 400   # Updated to match model_train.py
IMAGE_CHANNELS = 1  # Changed from 3 to 1 for Grayscale

# Global variables for normalization (to be loaded from stats.json)
global_mean = 0.0
global_std = 1.0

def load_global_statistics(stats_path):
    """
    Load global statistics for signal normalization.
    """
    global global_mean, global_std
    if not os.path.exists(stats_path):
        logging.error(f"Statistics file not found: {stats_path}")
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    global_mean = stats.get('mean', 0.0)
    global_std = stats.get('std', 1.0)
    logging.info(f"Loaded Global Mean: {global_mean}, Global Std: {global_std}")

def paired_data_generator(image_dir, signal_dir, mode='train'):
    """
    Generator that yields paired image and signal data, along with the corresponding label (signal).
    Each signal is paired with its corresponding image.
    
    Args:
        image_dir (str): Directory containing image files.
        signal_dir (str): Directory containing signal files.
        mode (str): 'train', 'validation', or 'test' to handle different input requirements.
    """
    signal_files = sorted(Path(signal_dir).glob('*.dat'))
    missing_images = []
    total_files = len(signal_files)
    logging.info(f"Found {total_files} signal files in {signal_dir}")

    for sig in signal_files:
        # Extract the stem without suffix
        stem = sig.stem  # e.g., '00940_lr' from '00940_lr.dat'

        # Construct the corresponding image filename
        image_filename = f"{stem}.png"
        image_path = Path(image_dir) / image_filename

        if image_path.exists():
            # Load image
            image = load_image(str(image_path))
            # Load signal
            signal = load_signal(str(sig.with_suffix('')))  # Remove .dat extension
            
            # Yield ((image, signal), signal) pair as expected by TensorFlow
            yield (image, signal), signal
        else:
            missing_images.append(str(sig))
            logging.warning(f"No corresponding image found for signal {sig}. Skipping.")

    if missing_images:
        logging.info(f"Total missing images: {len(missing_images)} out of {total_files} signals.")

def load_image(image_path):
    """
    Load and preprocess the image.
    """
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=IMAGE_CHANNELS)  # Ensure single channel
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0,1]
        image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH], method='bilinear')  # Ensure resizing
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        # Return a zero image or handle as per requirements
        return tf.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32)


def load_signal(signal_path):
    """
    Load and preprocess the signal.
    """
    try:
        record = wfdb.rdrecord(signal_path)
        signal = record.p_signal.flatten()  # Flatten if multichannel
    except Exception as e:
        logging.error(f"Error loading signal {signal_path}: {e}")
        # Return a zero signal or handle as per requirements
        signal = np.zeros(SIGNAL_INPUT_LENGTH, dtype=np.float32)
    
    # Normalize the signal based on global statistics
    if global_std != 0:
        signal = (signal - global_mean) / global_std
    else:
        logging.warning("Global std is zero. Skipping normalization.")
        signal = signal - global_mean
    
    # Truncate or pad the signal to SIGNAL_INPUT_LENGTH
    if len(signal) > SIGNAL_INPUT_LENGTH:
        signal = signal[:SIGNAL_INPUT_LENGTH]
    else:
        signal = np.pad(signal, (0, SIGNAL_INPUT_LENGTH - len(signal)), 'constant')
    
    # Clip the signal to prevent extreme values
    signal = np.clip(signal, -5.0, 5.0)
    
    # Check for NaNs
    if np.isnan(signal).any():
        logging.warning(f"NaN values found in signal {signal_path}. Replacing with zeros.")
        signal = np.nan_to_num(signal)
    
    signal = tf.convert_to_tensor(signal, dtype=tf.float32)
    return signal

def augment(inputs, label):
    """
    Apply data augmentation to images and signals.
    
    Args:
        inputs (tuple): ((image, signal), label)
        label (tensor): Reconstructed signal (signal)
    Returns:
        Tuple: ((augmented_image, augmented_signal), label)
    """
    (image, signal), label = inputs, label

    # Image Augmentations (compatible with Grayscale)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Removed saturation and hue adjustments for grayscale images

    # Signal Augmentations
    noise = tf.random.normal(shape=tf.shape(signal), mean=0.0, stddev=0.01)
    signal = signal + noise
    signal = tf.roll(signal, shift=100, axis=0)
    signal = signal * 1.1

    # Clip the signal to prevent large values
    signal = tf.clip_by_value(signal, -5.0, 5.0)

    # Replace any NaNs resulting from augmentation
    signal = tf.where(tf.math.is_nan(signal), tf.zeros_like(signal), signal)

    return ((image, signal), label)

def create_dataset(image_dir, signal_dir, batch_size=16, shuffle=True, augment_data=False, repeat=False, mode='train'):
    """
    Create a TensorFlow dataset from paired image and signal data.
    
    Args:
        image_dir (str): Directory containing image files.
        signal_dir (str): Directory containing signal files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        augment_data (bool): Whether to apply data augmentation.
        repeat (bool): Whether to repeat the dataset indefinitely.
        mode (str): Mode of the dataset ('train', 'validation', 'test').

    Returns:
        tf.data.Dataset: Prepared TensorFlow dataset.
    """
    if mode not in ['train', 'validation', 'test']:
        raise ValueError(f"Unknown mode: {mode}. Use 'train', 'validation', or 'test'.")

    # Define output signature based on mode
    output_signature = (
        (
            tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(SIGNAL_INPUT_LENGTH,), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(SIGNAL_INPUT_LENGTH,), dtype=tf.float32)  # Expecting signal as label
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: paired_data_generator(image_dir, signal_dir, mode=mode),
        output_signature=output_signature
    )

    # Shuffling only for training
    if shuffle and mode == 'train':
        total_signals = len(list(Path(signal_dir).glob('*.dat')))
        if total_signals == 0:
            logging.warning(f"No signal files found in {signal_dir} to shuffle.")
            buffer_size = 1
        else:
            buffer_size = min(1000, total_signals)
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Repeating only for training
    if repeat and mode == 'train':
        dataset = dataset.repeat()

    # Apply augmentation only for training
    if augment_data and mode == 'train':
        dataset = dataset.map(lambda inputs, label: (augment(inputs, label)), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the data
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    # Example usage:
    # python data_loader.py --image-dir /path/to/images --signal-dir /path/to/signals --stats-path /path/to/stats.json
    parser = argparse.ArgumentParser(description='Data Loader for ECG Reconstruction Model.')
    parser.add_argument('--image-dir', required=True, help='Directory containing image files.')
    parser.add_argument('--signal-dir', required=True, help='Directory containing signal files.')
    parser.add_argument('--stats-path', required=True, help='Path to the stats.json file for normalization.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load global statistics
    load_global_statistics(args.stats_path)

    # Create dataset (example parameters)
    dataset = create_dataset(
        image_dir=args.image_dir,
        signal_dir=args.signal_dir,
        batch_size=16,
        shuffle=True,
        augment_data=True,  # Set to True to apply augmentation
        repeat=True,
        mode='train'
    )

    # Iterate through the dataset
    for (inputs, labels) in dataset.take(1):
        images = inputs['image_input']
        signals = inputs['signal_input']
        logging.info(f"Batch Images Shape: {images.shape}")
        logging.info(f"Batch Signals Shape: {signals.shape}")
        logging.info(f"Batch Labels Shape: {labels.shape}")
