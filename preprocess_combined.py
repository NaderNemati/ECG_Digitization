import argparse
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import wfdb
from scipy.signal import butter, sosfilt, iirnotch, filtfilt, find_peaks
import biosppy
import json
from scipy.ndimage import gaussian_filter

# Constants for preprocessing
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
SIGNAL_EXTENSION = '.dat'
IMAGE_PLOT_DIR = '/home/nader/Desktop/ECG_Digitalization-main/ptb-xl/preprocessed_data/img_plot'
SIGNAL_PLOT_DIR = '/home/nader/Desktop/ECG_Digitalization-main/ptb-xl/preprocessed_data/signal_plot'
os.makedirs(IMAGE_PLOT_DIR, exist_ok=True)
os.makedirs(SIGNAL_PLOT_DIR, exist_ok=True)

# Configure Logging
logging.basicConfig(
    filename='preprocess_ecg_combined.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants for signal preprocessing
LOW_CUT = 0.5  # Hz
HIGH_CUT = 40.0  # Hz
SAMPLING_RATE = 500.0  # Hz
NOTCH_FREQ = 50.0  # Hz
NOTCH_QUALITY_FACTOR = 25.0
SMOOTHING_SIGMA = 1  # Sigma for Gaussian smoothing
SIGNAL_INPUT_LENGTH = 10000  # Samples

# Global statistics for normalization
global_mean, global_std = None, None

def visualize_image(image, title, filename, plot_dir=IMAGE_PLOT_DIR):
    """
    Save visualization plots for images.
    """
    plt.figure(figsize=(10, 8))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.close()

def visualize_signal(original_signal, preprocessed_signal, peaks, down_peaks, qrs_complexes, title, filename, plot_dir=SIGNAL_PLOT_DIR):
    """
    Save visualization plots for signals.
    """
    plt.figure(figsize=(12, 6))
    if original_signal is not None:
        plt.plot(original_signal, label='Original ECG Signal', color='blue', alpha=0.5)
    plt.plot(preprocessed_signal, label='Preprocessed ECG Signal', color='green')
    if peaks is not None:
        plt.scatter(peaks, preprocessed_signal[peaks], color='red', label='R-peaks')
    if down_peaks is not None:
        plt.scatter(down_peaks, preprocessed_signal[down_peaks], color='orange', label='Down Peaks')
    for start, end in qrs_complexes:
        plt.axvspan(start, end, color="yellow", alpha=0.3, label="QRS Complex" if start == qrs_complexes[0][0] else "")
    plt.title(title)
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError(f"Filter frequencies must be in the range 0 < Wn < 1. Got low: {low}, high: {high}.")

    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=SAMPLING_RATE, order=4):
    """Apply a bandpass filter to the data."""
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    return sosfilt(sos, data)

def notch_filter(data, fs=SAMPLING_RATE, freq=NOTCH_FREQ, Q=NOTCH_QUALITY_FACTOR):
    """
    Apply a notch filter to remove a specific frequency from the signal.

    Parameters:
    - data (array-like): The input signal.
    - fs (float): Sampling frequency in Hz.
    - freq (float): The frequency to be removed from the signal in Hz.
    - Q (float): Quality factor.

    Returns:
    - filtered_data (array-like): The notch-filtered signal.
    """
    try:
        nyquist = 0.5 * fs
        notch_freq = freq / nyquist
        logging.info(f"Applying notch filter: freq={freq}Hz, Q={Q}, fs={fs}Hz")
        if not (0 < notch_freq < 1):
            raise ValueError(f"Notch frequency {freq} Hz is out of valid range for the given sampling rate {fs} Hz.")
        b, a = iirnotch(notch_freq, Q)  # Pass Q as positional argument
        filtered_data = filtfilt(b, a, data)
        logging.info("Notch filter applied successfully.")
        return filtered_data
    except Exception as e:
        logging.error(f"Notch filter failed: {e}")
        # Optionally, return the original data or handle as needed
        return data

def smooth_signal(signal, sigma=SMOOTHING_SIGMA):
    """Apply Gaussian smoothing to the signal."""
    return gaussian_filter(signal, sigma=sigma)

def global_z_score_normalize(signals):
    """Normalize signals using global mean and std."""
    epsilon = 1e-8  # Small constant to prevent division by zero
    if global_std == 0:
        logging.warning("Global standard deviation is zero, skipping normalization.")
        return signals
    norm_signals = (signals - global_mean) / (global_std + epsilon)
    return norm_signals

def detect_qrs_pan_tompkins(signal, fs):
    """Detect QRS complexes using the Pan-Tompkins algorithm via biosppy."""
    try:
        qrs = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        qrs_indices = qrs['rpeaks']
        logging.info(f"Detected {len(qrs_indices)} QRS complexes using Pan-Tompkins.")
        return qrs_indices
    except Exception as e:
        logging.error(f"Pan-Tompkins QRS detection failed: {e}")
        return np.array([])

def detect_down_peaks(signal, distance=200, prominence=0.6):
    """Detect down peaks (e.g., S-peaks or T-peaks) in the ECG signal."""
    inverted_signal = -signal
    down_peaks, properties = find_peaks(inverted_signal, distance=distance, prominence=prominence)
    logging.info(f"Detected {len(down_peaks)} down peaks.")
    return down_peaks, properties

def detect_qrs_complex(signal, peaks, window=50):
    """Detect the QRS complex around the R-peaks."""
    qrs_complexes = []
    for peak in peaks:
        qrs_start = max(peak - window, 0)
        qrs_end = min(peak + window, len(signal))
        qrs_complexes.append((qrs_start, qrs_end))
    return qrs_complexes

def calculate_rr_intervals(peaks):
    """Calculate RR intervals from detected peaks (R-peaks)."""
    if len(peaks) < 2:
        return np.array([])
    rr_intervals = np.diff(peaks) / SAMPLING_RATE  # Convert samples to seconds
    return rr_intervals

def preprocess_signal(file_prefix, apply_notch=True, apply_bandpass=True, apply_smoothing=True, display=False):
    """Preprocess a single ECG signal file by processing only the 2nd lead (Lead II)."""
    try:
        # Read only the 2nd lead (Lead II, assuming 0-based indexing)
        record = wfdb.rdrecord(file_prefix, channels=[1])  # Read only Lead II
        logging.info(f"Successfully read signal file: {file_prefix}")
    except Exception as e:
        logging.error(f"Failed to read signal file {file_prefix}: {e}")
        return None, None, None, None, None, None, None

    # Extract Lead II
    signals = record.p_signal  # Shape: (n_samples, 1)
    if signals.ndim != 2 or signals.shape[1] != 1:
        logging.error(f"Unexpected signal shape for {file_prefix}: {signals.shape}. Expected (n_samples, 1).")
        return None, None, None, None, None, None, None

    # Handle NaNs by replacing them with zero
    if np.isnan(signals).any():
        signals = np.nan_to_num(signals)

    # Process only Lead II
    processed_signals = []
    for lead in range(1):  # Only one lead
        signal = signals[:, lead]

        if apply_notch:
            # Commented out notch filter
            # signal = notch_filter(signal, record.fs, freq=NOTCH_FREQ, Q=NOTCH_QUALITY_FACTOR)
            pass  # Notch filter is disabled

        if apply_bandpass:
            # Commented out band-pass filter
            # signal = bandpass_filter(signal, lowcut=LOW_CUT, highcut=HIGH_CUT, fs=record.fs, order=4)
            pass  # Band-pass filter is disabled

        if apply_smoothing:
            signal = smooth_signal(signal, sigma=SMOOTHING_SIGMA)

        # Pad or truncate to SIGNAL_INPUT_LENGTH
        if len(signal) < SIGNAL_INPUT_LENGTH:
            padding_length = SIGNAL_INPUT_LENGTH - len(signal)
            signal = np.pad(signal, (0, padding_length), 'reflect')
        else:
            signal = signal[:SIGNAL_INPUT_LENGTH]

        # Ensure consistent signal length
        assert len(signal) == SIGNAL_INPUT_LENGTH, f"Signal length mismatch: {len(signal)} vs {SIGNAL_INPUT_LENGTH}"

        processed_signals.append(signal)

    # Normalize the processed signal
    processed_signals = np.array(processed_signals)  # Shape: (1, SIGNAL_INPUT_LENGTH)
    # Commented out normalization
    # processed_signals = global_z_score_normalize(processed_signals)

    # Extract Lead II's signal for QRS detection and visualization
    label_signal = processed_signals[0]

    # Check for flat signal
    if np.all(label_signal == label_signal[0]):
        logging.warning(f"Signal {file_prefix} is flat. Skipping peak detection.")
        return processed_signals, None, None, None, None, record.fs, None

    # Detect QRS complexes using Pan-Tompkins
    qrs_peaks = detect_qrs_pan_tompkins(label_signal, record.fs)

    # Detect down peaks
    down_peaks, down_peak_props = detect_down_peaks(label_signal)

    # If Pan-Tompkins fails, fallback to general peak detection
    if len(qrs_peaks) == 0:
        logging.warning(f"Pan-Tompkins failed for {file_prefix}. Falling back to general peak detection.")
        # General peak detection parameters
        min_distance = int(0.4 * record.fs)  # Minimum 0.4 seconds between peaks
        qrs_peaks, _ = find_peaks(label_signal, distance=min_distance, prominence=0.6)
        logging.info(f"Detected {len(qrs_peaks)} QRS complexes using general peak detection.")

    qrs_complexes = detect_qrs_complex(label_signal, qrs_peaks)
    rr_intervals = calculate_rr_intervals(qrs_peaks)

    # Visualization: Preprocessed Signal with Detected Peaks (only for Lead II)
    if display:
        visualize_signal(
            original_signal=None,  # Original signal before preprocessing is not available here
            preprocessed_signal=label_signal,
            peaks=qrs_peaks,
            down_peaks=down_peaks,
            qrs_complexes=qrs_complexes,
            title=f"Preprocessed Signal with Peaks for {file_prefix} (Lead II)",
            filename=f"{Path(file_prefix).stem}_peaks.png",
            plot_dir=SIGNAL_PLOT_DIR
        )

    logging.info(f"Successfully processed signal for {file_prefix}: {len(qrs_peaks)} QRS peaks detected.")
    return processed_signals, qrs_peaks, qrs_complexes, rr_intervals, down_peaks, record.fs, None

def preprocess_image(file_path, display=False, filename=""):
    """Preprocess a single ECG image."""
    image = cv2.imread(str(file_path))
    if image is None:
        logging.error(f"Unable to read image: {file_path}. Skipping.")
        return None

    # Convert to grayscale if not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Correct skew
    corrected_image = correct_skew(image, display=display, filename=filename)

    # Enhance contrast
    contrast_enhanced = enhance_contrast(corrected_image, display=display, filename=filename)

    # Remove large grid lines
    gridless_image = remove_large_grid_lines(contrast_enhanced, display=display, filename=filename)

    # Denoise the image
    denoised_image = denoise_image(gridless_image, display=display, filename=filename)

    # Normalize the image size while maintaining aspect ratio
    normalized_image = normalize_image(denoised_image, size=(800, 600), display=display, filename=filename)

    return normalized_image

def correct_skew(image, display=False, filename=""):
    """
    Correct the skew of an image using Hough Line Transform.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Probabilistic Hough Transform for better accuracy
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

        if len(angles) == 0:
            logging.warning(f"No angles detected for skew correction in {filename}. Skipping rotation.")
            return image

        median_angle = np.median(angles)

        # Clamp the median_angle to prevent over-rotation
        max_rotation = 0  # Maximum degrees to rotate
        median_angle = max(min(median_angle, max_rotation), -max_rotation)

        if np.abs(median_angle) > 1:  # Rotate only if skew angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated_image = cv2.warpAffine(
                image,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            if display:
                logging.info(f"Corrected skew by {median_angle:.2f} degrees for {filename}.")
                visualize_image(
                    rotated_image,
                    title=f"Skew Corrected ({median_angle:.2f}°)",
                    filename=f"skew_corrected_{filename}.png",
                    plot_dir=IMAGE_PLOT_DIR
                )

            return rotated_image
        else:
            logging.info(f"No significant skew detected for {filename}. Skipping rotation.")
            return image
    else:
        logging.warning(f"No lines detected for skew correction in {filename}. Skipping rotation.")
        return image

def enhance_contrast(image, display=False, filename=""):
    """
    Enhance the contrast of an image using histogram equalization.
    """
    if len(image.shape) == 2:
        # Grayscale image
        equalized = cv2.equalizeHist(image)
    else:
        # Color image
        img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr, cb = cv2.split(img_y_cr_cb)
        y_channel_eq = cv2.equalizeHist(y_channel)
        img_y_cr_cb_eq = cv2.merge([y_channel_eq, cr, cb])
        equalized = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)

    if display:
        visualize_image(
            equalized,
            title="Contrast Enhanced",
            filename=f"contrast_enhanced_{filename}.png",
            plot_dir=IMAGE_PLOT_DIR
        )

    return equalized

def remove_large_grid_lines(image, display=False, filename=""):
    """
    Remove large grid lines from an ECG image using adaptive thresholding and morphological operations.
    """
    # Adaptive Thresholding to create binary image
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )

    # Define kernels for detecting horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine detected lines to create a grid mask
    grid_mask = cv2.add(horizontal_lines, vertical_lines)

    # Inpaint the original image using the grid mask
    inpaint_radius = 3
    if len(image.shape) == 2:
        inpainted = cv2.inpaint(image, grid_mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        inpainted = cv2.inpaint(image, grid_mask, inpaint_radius, cv2.INPAINT_TELEA)

    if display:
        visualize_image(
            inpainted,
            title="Grid Lines Removed",
            filename=f"grid_removed_{filename}.png",
            plot_dir=IMAGE_PLOT_DIR
        )

    return inpainted

def denoise_image(image, display=False, filename=""):
    """
    Apply denoising techniques to smooth the image while preserving edges.
    """
    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)

    if display:
        visualize_image(
            denoised,
            title="Denoised Image",
            filename=f"denoised_{filename}.png",
            plot_dir=IMAGE_PLOT_DIR
        )

    return denoised

def normalize_image(image, size=(800, 600), display=False, filename=""):
    """
    Resize and normalize the image to a standard size while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    desired_w, desired_h = size
    aspect_ratio = w / h
    desired_aspect = desired_w / desired_h

    if aspect_ratio > desired_aspect:
        # Image is wider than desired aspect ratio
        new_w = desired_w
        new_h = int(desired_w / aspect_ratio)
    else:
        # Image is taller than desired aspect ratio
        new_h = desired_h
        new_w = int(desired_h * aspect_ratio)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas with desired size
    if len(image.shape) == 2:
        canvas = np.full((desired_h, desired_w), 255, dtype=np.uint8)  # White background
    else:
        canvas = np.full((desired_h, desired_w, 3), 255, dtype=np.uint8)  # White background for color images

    # Compute top-left corner for centered placement
    x_offset = (desired_w - new_w) // 2
    y_offset = (desired_h - new_h) // 2

    # Place the resized image onto the canvas
    if len(image.shape) == 2:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    if display:
        visualize_image(
            canvas,
            title="Normalized Image",
            filename=f"normalized_{filename}.png",
            plot_dir=IMAGE_PLOT_DIR
        )

    return canvas

def preprocess_and_save_combined_data(image_file, signal_file, output_folder, split_type, display=False):
    """
    Preprocess and save paired ECG image and signal data.

    Parameters:
    - image_file (Path): Path to the ECG image file.
    - signal_file (Path): Path to the corresponding raw ECG signal file (.dat).
    - output_folder (str): Base directory to save preprocessed data (train/test).
    - split_type (str): 'train' or 'test' to determine the target folder.
    - display (bool): Whether to generate and save intermediate visualizations.

    Returns:
    - bool: True if both image and signal are processed and saved successfully, False otherwise.
    """
    filename = signal_file.stem  # Use signal stem for naming

    # Preprocess Image
    preprocessed_image = preprocess_image(image_file, display=display, filename=filename)
    if preprocessed_image is None:
        logging.error(f"Image preprocessing failed for {image_file}.")
        return False

    # Preprocess Signal (Only Lead II)
    processed_signals, qrs_peaks, qrs_complexes, rr_intervals, down_peaks, fs, _ = preprocess_signal(str(signal_file).replace(SIGNAL_EXTENSION, ''), display=display)
    if processed_signals is None:
        logging.error(f"Signal preprocessing failed for {signal_file}.")
        return False

    # Determine target folders
    images_folder = os.path.join(output_folder, split_type, 'images')
    signals_folder = os.path.join(output_folder, split_type, 'signals')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(signals_folder, exist_ok=True)

    # Save Image
    image_save_path = os.path.join(images_folder, f"{filename}.png")
    success_image = cv2.imwrite(image_save_path, preprocessed_image)
    if not success_image:
        logging.error(f"Failed to save preprocessed image: {image_save_path}")
        return False

    # Save Only Lead II Signal using WFDB
    # Transpose to shape (n_samples, 1)
    p_signal = processed_signals.T  # Shape: (SIGNAL_INPUT_LENGTH, 1)
    try:
        wfdb.wrsamp(
            record_name=filename,
            fs=fs,
            units=['mV'],  # Only one lead
            sig_name=['II'],  # Lead II
            p_signal=p_signal,
            write_dir=signals_folder
        )
    except Exception as e:
        logging.error(f"Failed to save signal for {filename}: {e}")
        return False

    # Save Visualization of Signal (Lead II)
    if qrs_peaks is not None and down_peaks is not None:
        visualize_signal(
            original_signal=None,  # Original signal before preprocessing is not available here
            preprocessed_signal=processed_signals[0],  # Lead II
            peaks=qrs_peaks,
            down_peaks=down_peaks,
            qrs_complexes=qrs_complexes,
            title=f"Preprocessed Signal with Peaks for {filename} (Lead II)",
            filename=f"{filename}_peaks.png",
            plot_dir=SIGNAL_PLOT_DIR
        )

    logging.info(f"Successfully processed and saved image and signal for {filename}.")
    return True

def calculate_global_statistics(input_signal_folder, stats_save_path):
    """
    Calculate the global mean and std for normalization incrementally and save them.
    Only computes statistics for Lead II.

    Parameters:
    - input_signal_folder (str): Directory containing preprocessed signal files.
    - stats_save_path (str): Path to save the global statistics JSON file.
    """
    global global_mean, global_std
    signal_files = sorted(Path(input_signal_folder).glob(f'*{SIGNAL_EXTENSION}'))

    count = 0
    sum_signal = 0.0
    sum_sq_signal = 0.0

    for file in signal_files:
        try:
            # Read only Lead II
            record = wfdb.rdrecord(str(file).replace(SIGNAL_EXTENSION, ''), channels=[1])  # Read only Lead II
            signal = record.p_signal  # Shape: (n_samples, 1)
            signal = np.nan_to_num(signal)  # Replace NaNs with zero

            # Pad or truncate to SIGNAL_INPUT_LENGTH
            if signal.shape[0] < SIGNAL_INPUT_LENGTH:
                padding_length = SIGNAL_INPUT_LENGTH - signal.shape[0]
                signal = np.pad(signal, ((0, padding_length), (0, 0)), 'reflect')
            else:
                signal = signal[:SIGNAL_INPUT_LENGTH, :]

            # Flatten the signal to 1D
            signal_flat = signal.flatten()

            # Update count, sum, and sum of squares
            count += signal_flat.size
            sum_signal += np.sum(signal_flat)
            sum_sq_signal += np.sum(signal_flat ** 2)
        except Exception as e:
            logging.error(f"Failed to read signal file {file}: {e}")

    if count > 0:
        global_mean = sum_signal / count
        global_std = np.sqrt((sum_sq_signal / count) - (global_mean ** 2))
    else:
        global_mean, global_std = 0, 1  # Set defaults if no signals are available

    # Save global statistics to a JSON file
    stats = {
        'mean': global_mean,
        'std': global_std
    }
    with open(stats_save_path, 'w') as f:
        json.dump(stats, f)

    logging.info(f"Global Mean: {global_mean}, Global Std: {global_std}")
    print(f"Global Mean: {global_mean}, Global Std: {global_std}")

def preprocess_and_split_data(image_folder, signal_folder, output_folder, test_size=0.2, display=False):
    """
    Preprocess all paired ECG image and signal data and split into train/test sets.
    Only processes Lead II.

    Parameters:
    - image_folder (Path): Directory containing ECG image files.
    - signal_folder (Path): Directory containing raw ECG signal files.
    - output_folder (str): Base directory to save preprocessed data (train/test).
    - test_size (float): Proportion of data to allocate to the test set.
    - display (bool): Whether to generate and save intermediate visualizations.

    Returns:
    - None
    """
    os.makedirs(output_folder, exist_ok=True)
    train_images_folder = os.path.join(output_folder, 'train', 'images')
    train_signals_folder = os.path.join(output_folder, 'train', 'signals')
    test_images_folder = os.path.join(output_folder, 'test', 'images')
    test_signals_folder = os.path.join(output_folder, 'test', 'signals')
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_signals_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)
    os.makedirs(test_signals_folder, exist_ok=True)

    # Calculate global statistics for normalization and save them
    stats_save_path = os.path.join(output_folder, 'stats.json')
    calculate_global_statistics(signal_folder, stats_save_path)

    # Gather all signal files
    signal_files = sorted(Path(signal_folder).glob(f'*{SIGNAL_EXTENSION}'))

    # Gather all image files
    image_files = sorted(Path(image_folder).glob('*-0.png'))  # Assuming images have '-0.png' suffix

    # Create a mapping of signal stems to image files
    data_pairs = {}
    for signal in signal_files:
        stem = signal.stem  # e.g., '00001_lr'
        # Find image files that start with the stem and have '-0.png'
        image_pattern = f"{stem}-0.png"
        image_path = signal.parent / image_pattern
        if image_path.exists():
            data_pairs[stem] = {'image': image_path, 'signal': signal}
        else:
            logging.warning(f"No corresponding image found for signal {signal}. Skipping.")

    # Remove pairs with missing image or signal
    data_pairs = {k: v for k, v in data_pairs.items() if v['image'] is not None and v['signal'] is not None}

    # Check if any paired data exists
    if not data_pairs:
        raise ValueError("No paired image and signal data found. Please check the filename consistency.")

    # Split into train and test
    stems = list(data_pairs.keys())
    train_stems, test_stems = train_test_split(stems, test_size=test_size, random_state=42)

    # Process and save training data
    print(f"Processing {len(train_stems)} training samples...")
    for idx, stem in enumerate(train_stems, 1):
        image_file = data_pairs[stem]['image']
        signal_file = data_pairs[stem]['signal']
        success = preprocess_and_save_combined_data(image_file, signal_file, output_folder, split_type='train', display=display)
        if not success:
            logging.error(f"Failed to process and save data for {stem}.")
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(train_stems)} training samples...")

    # Process and save testing data
    print(f"Processing {len(test_stems)} testing samples...")
    for idx, stem in enumerate(test_stems, 1):
        image_file = data_pairs[stem]['image']
        signal_file = data_pairs[stem]['signal']
        success = preprocess_and_save_combined_data(image_file, signal_file, output_folder, split_type='test', display=display)
        if not success:
            logging.error(f"Failed to process and save data for {stem}.")
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(test_stems)} testing samples...")

    logging.info("Preprocessing and splitting completed successfully.")
    print("Preprocessing and splitting completed successfully. Check the log file for details.")

def main():
    """Main function to parse arguments and start preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess and split paired ECG images and signals into train/test sets. Only Lead II is used.')
    parser.add_argument('--input-folder', required=True, help='Folder containing raw ECG records and images (e.g., /home/nader/Desktop/ECG_Digitalization-main/ptb-xl/records100_hidden/00000)')
    parser.add_argument('--output-folder', required=True, help='Base folder to save preprocessed data (train/test)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of data for the test set (default: 0.2)')
    parser.add_argument('--display', action='store_true', help='Enable visualization of preprocessing steps')
    args = parser.parse_args()

    image_folder = Path(args.input_folder)
    signal_folder = Path(args.input_folder)  # Assuming all files are in the same directory

    preprocess_and_split_data(
        image_folder=image_folder,
        signal_folder=signal_folder,
        output_folder=args.output_folder,
        test_size=args.test_size,
        display=args.display
    )

if __name__ == '__main__':
    main()
