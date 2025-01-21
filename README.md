# **Model Training Script (`model_train.py`)**

## **Overview**
The `model_train.py` script is a **multi-input deep learning training pipeline** designed for **ECG signal reconstruction** using paired **ECG images and signals**. It builds and trains a **hybrid model**, incorporating both **CNN-based image encoding** and **dense-based signal encoding** to reconstruct digital ECG signals from input images and corresponding signals.

## **Key Features**
### **1. Preprocessing Compatibility**
- The model processes both **ECG images and signals**, ensuring:
  - **Image Size:** `(400x400, RGB)`
  - **Signal Length:** `5000 samples`

### **2. Multi-Input Model Architecture**
- **Image Encoder:**
  - Uses either a **CNN-based encoder** or a **ViT-based transformer encoder**.
- **Signal Encoder:**
  - Fully connected dense layers for ECG feature extraction.
- **Decoder:**
  - Reconstructs ECG signals from the combined latent representations.

### **3. Custom Metrics for Evaluation**
- Implements **Signal-to-Noise Ratio (SNR)** as a **TensorFlow custom metric** to assess signal reconstruction quality.

### **4. Efficient Training Pipeline**
- Uses **TensorFlow's Dataset API** for fast data loading and preprocessing.
- Supports **learning rate scheduling, early stopping, and checkpointing**.
- Employs **`AdamW` optimizer with gradient clipping** to prevent exploding gradients.

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
