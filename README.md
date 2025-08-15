
# 1D CNN-LSTM for Seizure Detection (EEG Signals)

This project implements a deep learning-based model for epileptic seizure detection using EEG signals. The model employs **Continuous Wavelet Transform (CWT)** as a feature extractor and a combination of **1D CNN** and **LSTM** (Long Short-Term Memory) networks as the classifier.

## Table of Contents
- [Description](#description)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Running the Code](#running-the-code)
- [Script Details](#script-details)

## Description

This project uses a **1D CNN-LSTM** model for detecting seizures from EEG signals. The features are extracted using the **Continuous Wavelet Transform (CWT)** method, and the model classifies the signals as either seizure or non-seizure events. The model is designed to handle datasets like **BONN**, **CHB-MIT**, and **TUSZ**.

## Installation Instructions

Follow the steps below to set up the project:

### 1. Create a Virtual Environment (Optional but recommended)

To isolate your project dependencies, create a new virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate  # For Linux or macOS
myenv\Scriptsctivate  # For Windows
```

### 2. Install Required Dependencies

Below are the necessary libraries and the corresponding versions, based on the code:

- Python 3.7+ (Ensure you are using Python 3.7 or later for compatibility with Keras and TensorFlow.)

For libraries:

```bash
pip install tensorflow==2.10.0 keras==2.10.0 pywt matplotlib scikit-learn mne pyedflib shap tqdm pandas
```

### 3. Specific Version of TensorFlow and Keras

Ensure that you are using the compatible versions of TensorFlow and Keras:

- **TensorFlow 2.10.0**
- **Keras 2.10.0**

These versions are compatible with the code provided.

### 4. Dependencies in the Code

- **pywt**: Used for wavelet transformation (already included).
- **mne**: Used for handling EEG data and reading EDF files.
- **pyedflib**: Used for working with EEG files in EDF format.
- **scikit-learn**: For machine learning utilities like train-test split, cross-validation, and metrics.
- **shap**: For SHapley additive explanations (model explainability).

### 5. Set up the Datasets

Ensure the datasets (**BONN**, **CHB-MIT**, **TUSZ**) are available and loaded properly as described in the functions.

If the datasets are stored in specific directories, you may need to update paths in the code to reflect the actual locations of your data files.

### 6. Run the Code

After setting up the environment and installing the necessary libraries, run the provided scripts:

- `Load_and_Preprocessing_Data.py`: Handles dataset loading and preprocessing.
- `Extract_Wavelet_Coefficients.py`: Extracts wavelet coefficients for EEG data.
- `Model_Training.py`: Contains the training process for your models.
- `Main.py`: Controls the flow of the entire pipeline and asks for user input regarding dataset and classification type.
- `Visualization.py`: Plots training results, including accuracy and loss.

## Usage

### Example

Once everything is set up, run the `Main.py` script to begin:

```bash
python Main.py
```

This script will:
1. Ask for user input to choose a dataset (**BONN**, **CHB-MIT**, or **TUSZ**).
2. Ask for the classification type (**binary** or **multi**).
3. Execute the training and evaluation process.

## Dependencies

- **Python 3.7+**
- **TensorFlow 2.10.0**
- **Keras 2.10.0**
- **pywt** (Wavelet transform)
- **mne** (EEG data handling)
- **pyedflib** (EDF files handling)
- **scikit-learn** (Machine learning utilities)
- **shap** (Model explainability)
- **matplotlib** (Plotting)
- **tqdm** (Progress bar)
- **pandas** (Data handling)

## Datasets

This project uses the following datasets:
- **BONN**
- **CHB-MIT**
- **TUSZ**

Ensure that these datasets are available and loaded correctly for training and testing.

## Running the Code

1. Set up your virtual environment and install the dependencies.
2. Update paths for the datasets if required.
3. Run the code as follows:
    ```bash
    python Main.py
    ```

This will guide you through the dataset selection and classification type process.

## Script Details

### `Baseline_models.py`
This script provides baseline machine learning models (SVM, MLP, CNN, KNN, and Gaussian Naive Bayes) for EEG signal classification. It performs the following tasks:
- Preprocessing the data using **StandardScaler**.
- Splitting the data into training and testing sets.
- Training the models and calculating their accuracy.

### `ML_Classifiers_Training_Validation.py`
This script defines the training process for multiple machine learning classifiers, including:
- **SVC** (Support Vector Classifier)
- **KNN** (K-Nearest Neighbors)
- **Random Forest** and **Gradient Boosting**
- **MLP** (Multi-Layer Perceptron)
- **Gaussian Naive Bayes**
It also calculates several evaluation metrics like accuracy, F1 score, Matthews Correlation Coefficient (MCC), and others. The script includes:
- **Cross-validation** using K-Fold.
- **ROC Curve** and evaluation of multi-class classification.

### `Time_Frequency_Domain_Analysis.py`
This script performs **Time-Frequency Domain** feature extraction using **Welch's Method** and calculates important statistical features, including:
- **Mean Frequency** and **Median Frequency**.
It processes the EEG data by extracting both time-domain and frequency-domain features, then splits the dataset into training and testing sets for model evaluation.

Each script plays an essential role in the EEG signal classification pipeline, from feature extraction and model training to evaluation and visualization.
