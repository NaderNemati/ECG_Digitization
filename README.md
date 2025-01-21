# **Model Training Script (`model_train.py`) – Multi-Modal ECG Digitization**

## **Overview**
The `model_train.py` script is a **multi-input deep learning training pipeline** designed for **ECG signal reconstruction** using paired **ECG images and signals**. This model follows a **multi-modal learning** approach, leveraging information from **both image and signal domains** to enhance the digitization of ECG signals. 

## **Multi-Modal Learning in ECG Digitization**
Traditional ECG analysis relies on **either raw signals** (from electrodes) or **visual ECG printouts** (from medical records). However, **this model integrates both** to improve digitization accuracy:

- **Image Input**: ECG printouts contain essential waveform patterns but may be **distorted by noise, artifacts, and grid lines**. The model learns to extract features from these images, focusing on the waveform structure.
- **Signal Input**: The corresponding raw ECG signals provide precise **electrical activity** information. These signals help **correct distortions in the image input**.
- **Fusion Strategy**: The model **jointly processes both modalities** (image + signal) to reconstruct an accurate **digitized ECG waveform** that aligns with real ECG signal characteristics.

This approach **enhances the quality of reconstructed signals**, making ECG digitization **more robust to artifacts, missing segments, and variations** in ECG print formats.

---

## **Key Features**
### **1. Multi-Modal Model Architecture**
- **Image Encoder**:
  - Uses either a **CNN-based encoder** or a **ViT-based transformer encoder** to extract features from ECG images.
- **Signal Encoder**:
  - Uses dense layers to learn representations from raw ECG signals.
- **Decoder**:
  - Combines information from both encoders to reconstruct the original ECG signal.

### **2. ECG Digitization Pipeline**
- **Input**:
  - Raw ECG signals (`.dat` format)
  - ECG images (`.png`, `400x400`)
- **Processing**:
  - **Extract latent features** from images and signals.
  - **Fuse features** in a joint representation.
  - **Reconstruct the ECG signal**, minimizing error from noise and distortions.
- **Output**:
  - **Reconstructed ECG waveform**, aligned with medical-grade ECG readings.

### **3. Custom Metrics for Signal Quality**
- Implements a **Signal-to-Noise Ratio (SNR)** metric to assess the quality of reconstructed signals.
- Optimizes for **low reconstruction error** using MSE and MAE.

### **4. Efficient Training Pipeline**
- Uses **TensorFlow's Dataset API** for fast data loading and preprocessing.
- Supports **learning rate scheduling, early stopping, and checkpointing**.
- Employs **AdamW optimizer with gradient clipping** to prevent exploding gradients.

### **5. Evaluation & Plotting**
- Computes and logs **loss (MSE), MAE, and SNR** during training and validation.
- Generates:
  - **Learning curves**
  - **Signal reconstruction comparison plots**
  - **Test metrics visualization**

### **6. Command-Line Interface (CLI) Support**
- The script supports command-line arguments such as:
  ```bash
  --batch-size, --epochs, --learning-rate, --model-save-path
