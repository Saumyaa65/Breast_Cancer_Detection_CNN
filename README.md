# Breast Cancer Detection using 1D Convolutional Neural Network (CNN)

This project develops a machine learning model for **breast cancer detection**, classifying tumors as either **malignant** (cancerous) or **benign** (non-cancerous). It utilizes a **1D Convolutional Neural Network (CNN)**, which is particularly effective for analyzing patterns in tabular or sequential feature data.

## Overview

The primary objective is to build a robust and accurate predictive model using the Breast Cancer Wisconsin (Diagnostic) dataset. By applying a 1D CNN, the project demonstrates a powerful deep learning approach to medical diagnosis, leveraging the network's ability to learn complex relationships within diagnostic features.

## Features

* **Binary Classification:** Classifies breast cancer cases into two categories: malignant or benign.
* **1D Convolutional Neural Network (CNN):** Employs `Conv1D` layers to process the diagnostic features, enabling the model to identify subtle yet crucial patterns that contribute to classification.
* **Batch Normalization:** Integrates Batch Normalization layers to improve the stability and accelerate the training of the neural network.
* **Dropout Regularization:** Includes dropout layers to prevent overfitting, enhancing the model's ability to generalize to new, unseen patient data.
* **Feature Scaling:** Utilizes `StandardScaler` to normalize the input features, which is essential for optimal performance of neural networks.
* **Performance Evaluation:** Assesses model accuracy using a confusion matrix and overall accuracy score, providing insights into true positives, true negatives, false positives, and false negatives.
* **Training Visualization:** Plots training and validation accuracy/loss over epochs to monitor model learning and convergence.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn (`datasets`, `metrics`, `train_test_split`, `StandardScaler`, `confusion_matrix`, `accuracy_score`)

## How It Works

The project follows a standard machine learning pipeline:

1.  **Data Loading and Preparation:**
    * The **Breast Cancer Wisconsin (Diagnostic) dataset** is loaded directly from Scikit-learn's `datasets` module. This dataset contains various diagnostic measurements for breast mass examinations.
    * The features (`x`) and target variable (`y`, indicating malignant or benign) are separated.
    * The data is split into training and testing sets.

2.  **Feature Scaling and Reshaping:**
    * Features are scaled using `StandardScaler` to normalize their range, ensuring no single feature disproportionately influences the model.
    * The 2D feature matrix is then reshaped into a 3D format (`(samples, features, 1)`) to be compatible with the `Conv1D` layers of the CNN.

3.  **Model Architecture Definition:**
    * A Sequential Keras model is constructed.
    * It includes multiple `Conv1D` layers, `BatchNormalization` layers, and `Dropout` layers for robust feature learning and regularization.
    * The output of the convolutional layers is `Flattened` and fed into `Dense` (fully connected) layers.
    * The final `Dense` layer has a single unit with a `sigmoid` activation function, suitable for binary classification (predicting the probability of malignancy).

4.  **Model Compilation and Training:**
    * The model is compiled with the `Adam` optimizer (using a low learning rate for precise weight updates) and `binary_crossentropy` as the loss function, which is standard for binary classification problems.
    * The model is then trained on the preprocessed training data for a specified number of epochs, with validation performed on the test set to monitor its performance on unseen data.

5.  **Prediction and Evaluation:**
    * After training, the model predicts the probability of malignancy for the test set. These probabilities are converted into binary class labels (0 or 1) using a 0.5 threshold.
    * A **confusion matrix** is generated to visualize correct and incorrect classifications for both malignant and benign cases.
    * The **overall accuracy score** on the test set is also calculated.

6.  **Performance Visualization:**
    * Plots are generated to illustrate the model's training and validation accuracy over epochs, as well as the training and validation loss. These plots are crucial for understanding the model's learning trajectory and identifying potential issues like overfitting.
