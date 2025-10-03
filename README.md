# Alzheimers_Disease_Prediction
Built a lightweight neural network with 3 convolutional layer and a fully connected dense layer for classifying the MRI brain images into 4 catagories i.e., Mild Demented, Very Mild Demented, Moderate Demented, Non Demented.
This project leverages the Mendeley Alzheimer’s Disease Image Dataset to build a deep learning pipeline for Alzheimer’s disease prediction. The dataset contains 6,400 MRI images categorized into 4 classes, and the pipeline includes data preprocessing, augmentation, balancing, and model evaluation.
# Alzheimer's Disease Prediction from MRI Scans

This project focuses on the early prediction of Alzheimer's disease by classifying brain MRI scans into four distinct stages. Two separate deep learning models were developed and trained on two different, well-known datasets: the **ADNI dataset** and the **Mendeley dataset**, achieving test accuracies of **98.12%** and **96.88%** respectively.

---

## Table of Contents
- [About The Project](#-about-the-project)
- [Datasets Used](#-datasets-used)
- [Methodology](#-methodology)
- [Model Architecture](#-model-architecture)
- [Results & Performance](#-results--performance)

---

## About The Project

The goal of this project is to create reliable models for the early detection and classification of Alzheimer's disease using brain MRI scans. The models categorize images into four stages:

1.  **Non-Demented**
2.  **Very Mild Dementia**
3.  **Mild Dementia**
4.  **Moderate Dementia**

By training on two different datasets, this project demonstrates a robust approach to handling variations in medical imaging data and addresses common challenges like class imbalance.

---

## Datasets Used

Two publicly available datasets were used to train and evaluate the models.

### 1. ADNI Dataset
* This dataset was sourced from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.
* It contained a significant class imbalance, particularly for the 'Moderate Dementia' class.
* The dataset was split into training (80%), validation (10%), and testing (10%) sets to ensure a robust evaluation.

### 2. Mendeley Dataset
* This dataset is another well-regarded collection of Alzheimer's MRI scans and also has a notable class imbalance.
* Similar to the ADNI dataset, it was split into training (80%), validation (10%), and testing (10%) sets.

---

## Methodology

A consistent machine learning workflow was applied to both datasets to ensure comparability.

### 1. Data Preprocessing
* **Stratified Splitting**: Both datasets were split using a stratified approach to maintain the original distribution of classes across the training, validation, and test sets.
* **Data Augmentation & Balancing**: To address the class imbalance, the training set for both datasets was balanced using a powerful data augmentation pipeline with **Albumentations**. This process included techniques like horizontal flipping, affine transformations, and random brightness/contrast adjustments to generate unique, new images for the minority classes.

### 2. Model Training
* A **Sequential Deep Learning Model** was built using TensorFlow/Keras for both datasets.
* The models were trained for 50 epochs with an **Early Stopping** callback. This callback monitored the validation accuracy and stopped the training process after 10 epochs with no improvement, restoring the best-performing model weights to prevent overfitting.

---

## Model Architecture

The same Convolutional Neural Network (CNN) architecture was used for both models to ensure a fair comparison of the datasets. The architecture included:

* **Convolutional Layers (`Conv2D`)**: To extract features from the MRI images.
* **Batch Normalization**: To stabilize and speed up the training process.
* **Max Pooling Layers (`MaxPooling2D`)**: To reduce the dimensionality of the feature maps.
* **Dropout Layer**: To prevent overfitting.
* **Flatten Layer**: To prepare the data for the final classification layers.
* **Dense Layers**: Fully connected layers for the final classification, with a `softmax` activation function for the output layer.

---

## Results & Performance

Both models achieved high accuracy on their respective test sets, demonstrating the effectiveness of the approach.

### ADNI Dataset Model:
* **Test Accuracy**: **98.12%**
* **Key Insight**: The model performed exceptionally well, with very high precision and recall for all classes, including the heavily augmented 'Moderate Dementia' class.

### Mendeley Dataset Model:
* **Test Accuracy**: **96.88%**
* **Key Insight**: This model also demonstrated strong predictive power, successfully classifying the different stages of dementia with high accuracy.

