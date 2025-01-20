import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import wfdb

def load_features(record_path):
    """Load signal and extracted features from a given record path."""
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.flatten()
    
    features_path = f"{record_path}_features.npz"
    if os.path.exists(features_path):
        features = np.load(features_path, allow_pickle=True)
        return signal, features
    else:
        print(f"No features found for {record_path}")
        return None, None

def plot_features(record_name, signal, features):
    """Plot ECG signal and features such as RR intervals and QRS complexes."""
    plt.figure(figsize=(12, 8))
    
    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.plot(signal, label="ECG Signal")
    plt.title(f"ECG Signal and Features for {record_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot RR intervals
    if "rr_intervals" in features:
        rr_intervals = features["rr_intervals"]
        plt.subplot(3, 1, 2)
        plt.plot(rr_intervals, 'o-', label="RR Intervals (s)")
        plt.xlabel("Interval Index")
        plt.ylabel("RR Interval (seconds)")
        plt.legend()
    
    # Plot QRS complexes
    if "peaks" in features:
        peaks = features["peaks"]
        plt.subplot(3, 1, 3)
        plt.plot(signal, label="ECG Signal")
        plt.plot(peaks, signal[peaks], 'rx', label="R-peaks")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

    # Define the base output directory for plots
    output_base_path = Path("/home/nader/Desktop/ECG_Digitalization-main/ptb-xl/preprocessed_signals")
    
    # Create the directory if it doesn't exist
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Save each plot for each record
    output_path = output_base_path / f"{record_name}_features_plot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved at {output_path}")

def verify_features(input_folder):
    """Load and verify feature extraction for each record in the input folder."""
    records = sorted(Path(input_folder).glob("*.dat"))
    
    for record_path in records:
        record_name = record_path.stem
        signal, features = load_features(str(record_path).replace('.dat', ''))
        
        if signal is not None and features is not None:
            plot_features(record_name, signal, features)
        else:
            print(f"Skipping plot generation for {record_name}")

if __name__ == "__main__":
    input_folder = "/home/nader/Desktop/ECG_Digitalization-main/ptb-xl/preprocessed_signals"
    verify_features(input_folder)
