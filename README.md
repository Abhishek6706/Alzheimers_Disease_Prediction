# Alzheimer's Disease Prediction using Lightweight CNN

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview
This project focuses on the early diagnosis and classification of Alzheimer's Disease (AD) using a **Lightweight Convolutional Neural Network (CNN)**. The model is designed to be computationally efficient while maintaining high accuracy, making it suitable for deployment in low-resource clinical environments.

The system classifies MRI brain scans into four distinct stages of dementia:
1. **Non-Demented**
2. **Very Mild Demented**
3. **Mild Demented**
4. **Moderate Demented**

To ensure robustness, the model was trained and rigorously evaluated on two separate, well-known datasets (**ADNI** and **Mendeley**), achieving state-of-the-art test accuracies of **98.12%** and **96.88%** respectively.

## Datasets Used

The project leverages two publicly available datasets to test the model's ability to generalize across different image distributions.

| Dataset | Source | Images | Resolution | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset A (ADNI)** | Mendeley Advanced (2025) / ADNI Derived | 6,399 | $176 \times 208$ | **98.12%** |
| **Dataset B (Mendeley)** | Mendeley Data (2023) | 6,400 | $200 \times 190$ | **96.88%** |

> **Note:** Both datasets originally suffered from severe class imbalance (e.g., *Moderate Demented* constituted only ~1% of data). This was addressed using a custom stratified augmentation pipeline.


## Methodology

[cite_start]A consistent machine learning workflow was applied to both datasets to ensure comparability and reproducibility[cite: 37, 240, 242].

### 1. Data Preprocessing & Balancing
* **Stratified Splitting:** Data was split into Training (80%), Validation (10%), and Testing (10%) sets using stratified sampling to preserve class distribution.
* **Augmentation Pipeline:** We utilized **Albumentations** and standard Keras preprocessing to generate synthetic data for minority classes. Techniques included:
    * Horizontal Flip ($p=0.5$)
    * [cite_start]Affine Transformations (Scaling, Translation, Shearing) 
    * [cite_start]Random Brightness & Contrast ($p=0.8$) 
    * [cite_start]Coarse Dropout / Cutout ($p=0.5$) 

### 2. Novelty: Hashing-Based Uniqueness Check
To prevent data leakage and overfitting caused by duplicate augmented images, a **SHA-256 Hashing Mechanism** was implemented. 
* Every generated image is hashed.
* If a hash collision occurs (duplicate image), the sample is discarded and regenerated.
* This ensures the model trains on strictly unique data points.

### 3. Model Architecture
The core is a lightweight Sequential CNN designed to minimize parameters without sacrificing feature extraction capabilities.

* **Input Layer:** Resized to $180 \times 180$ (Grayscale).
* **Convolutional Blocks (x3):** * Filters: 16 $\rightarrow$ 32 $\rightarrow$ 64
    * Kernel: $3 \times 3$
    * Activation: ReLU
    * Pooling: MaxPooling2D ($2 \times 2$)
    * **Batch Normalization:** Applied after convolutions to stabilize training.
* **Global Features:** Flatten Layer.
* **Dense Block:** Fully connected layer (128 neurons) with **Dropout (0.5)** to prevent overfitting.
* **Output Layer:** Dense (4 neurons) with `Softmax` activation.

## Results & Performance

The models were trained for 50 epochs using the Adam optimizer (LR=0.001) with an **Early Stopping** callback (patience=10) to prevent overfitting.

### 1. In-Dataset Performance
| Metric | ADNI Model | Mendeley Model |
| :--- | :--- | :--- |
| **Accuracy** | **98.12%** | **96.88%** |
| **Precision (Weighted)** | 98% | 97% |
| **Recall (Weighted)** | 98% | 97% |
| **F1-Score (Weighted)** | 98% | 97% |

### 2. Cross-Dataset Generalization
To test real-world robustness, we performed a bidirectional evaluation where a model trained on one dataset was tested on the *completely unseen* second dataset.

* **Train on ADNI $\rightarrow$ Test on Mendeley:** 98% Accuracy
* **Train on Mendeley $\rightarrow$ Test on ADNI:** 98% Accuracy

This confirms the model learns disease-specific biomarkers rather than dataset-specific noise.


