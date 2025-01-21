# **Multi-Modal ECG Digitization**

## **Overview**
The `model_train.py` script is a **multi-input deep learning training pipeline** designed for **ECG signal reconstruction** using paired **ECG images and signals**. This model follows a **multi-modal learning approach**, where **ECG waveforms from printed images are fused with their corresponding digital signals** to reconstruct an accurate and noise-free ECG.

## **🚀 The Power of Fusing ECG Signals and Images**
Traditional ECG analysis relies on **either raw signals** (from electrodes) or **visual ECG printouts** (from medical records). However, **each modality alone has limitations**:

- **ECG Images**:  
  - Contain **waveform shape** but may suffer from **grid artifacts, noise, and distortions**.  
  - Important in cases where **digital ECG signals are unavailable** (e.g., scanned medical records).
- **Raw ECG Signals**:  
  - Provide **precise electrical activity** but might be **missing important shape features** if preprocessing removes critical aspects.  

🔑 **By fusing these two sources (image + signal), the model enhances ECG reconstruction accuracy.**  
✅ **The image provides waveform structure**, helping the model **recover missing or distorted regions**.  
✅ **The signal ensures precision**, correcting any distortions in the image-based interpretation.  
✅ **Combining both improves digitization**, allowing the reconstruction of a **faithful, clinically meaningful ECG**.

---

## **Key Features**
### **1. Multi-Modal Model Architecture**
- **Image Encoder**:
  - Uses either a **CNN-based encoder** or a **ViT-based transformer encoder** to extract features from ECG images.
- **Signal Encoder**:
  - Uses fully connected dense layers to learn representations from raw ECG signals.
- **Fusion Layer**:
  - Combines **image and signal features** into a joint representation.
- **Decoder**:
  - Processes the fused representation to **reconstruct the ECG signal** with **improved accuracy**.

### **2. ECG Digitization Pipeline**
- **Input**:
  - Raw ECG signals (`.dat` format).
  - ECG images (`.png`, `400x400`).
- **Processing**:
  - **Extract latent features** from images and signals.
  - **Fuse features** in a joint representation.
  - **Reconstruct the ECG signal**, minimizing noise and distortions.
- **Output**:
  - **Digitized ECG waveform**, aligned with real-world ECG recordings.

### **3. Custom Metrics for Signal Quality**
- Implements a **Signal-to-Noise Ratio (SNR) metric** to assess the quality of reconstructed signals.
- Optimizes for **low reconstruction error** using **MSE and MAE**.

### **4. Efficient Training Pipeline**
- Uses **TensorFlow’s Dataset API** for efficient data loading and preprocessing.
- Supports **learning rate scheduling, early stopping, and checkpointing**.
- Employs **AdamW optimizer with gradient clipping** to prevent exploding gradients.

### **5. Evaluation & Plotting**
- Computes and logs **MSE, MAE, and SNR** during training and validation.
- Generates:
  - **Learning curves** 📈
  - **Signal reconstruction comparison plots** 🔄
  - **Test metrics visualization** 📊

### **6. Command-Line Interface (CLI) Support**
- The script supports command-line arguments such as:
  ```bash
  --batch-size, --epochs, --learning-rate, --model-save-path
