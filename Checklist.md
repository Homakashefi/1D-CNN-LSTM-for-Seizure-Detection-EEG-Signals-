
### Updated Checklist for Replicating Main Tables

1. **Raw Data Access Instructions**:
   - **BONN EEG Dataset**:
     - Download from: [Set A (Z.zip)](http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/Z.zip), [Set B (O.zip)](http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/O.zip), [Set C (N.zip)](http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/N.zip), [Set D (F.zip)](http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/F.zip), [Set E (S.zip)](http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/S.zip)
     - **Description**: This dataset includes five sets of EEG recordings used for seizure detection. Each set contains 100 samples.
     - **Data Details**: Sampling rate: 173.61 Hz. The data is stored in `.txt` format with each file containing 4096 samples.

   - **CHB-MIT Scalp EEG Database**:
     - Download from: [CHB-MIT Scalp EEG Database](https://archive.physionet.org/physiobank/database/chbmit/)
     - **Description**: This database includes pediatric EEG recordings with seizure annotations. Data is available in European Data Format (EDF).
     - **Data Details**: Contains 23 subjects with seizures, providing annotated seizure events.

   - **TUSZ EEG Dataset**:
     - Download from: [TUSZ Dataset](https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/)
     - **Access Instructions**: Fill out the access request form and receive login credentials to download the dataset. The dataset contains sEEG data with seizure annotations.

2. **Exact Hyper-Parameter Settings**:
   - **Model Training Parameters**:
     - Batch size: 60
     - Number of epochs: 300
   - **Wavelet Transformation**:
     - Level for wavelet decomposition: 3
   - **Classifier Settings**:
     - SVM, KNN, MLP, RF, etc., with their specific hyperparameters set within `binary_ML_Classifiers` and `Multi_ML_Classifiers`.
   - **Neural Network**:
     - Adam optimizer with learning rate: 0.0001.
     - LSTM and CNN architecture details in `Model_Training.py`.

3. **Random-Seed Control**:
   - Random seed is explicitly set in functions to ensure reproducibility, for example:
     - Random split for training and testing sets is controlled with `random_state=42`.
     - KFold cross-validation is used with `random_state=2`.

4. **Minimal “Run-All” Script**:
   - The `Main.py` script handles dataset selection, classifier type choice, model training, and evaluation. This script can serve as the "run-all" script. It integrates all functions such as model training, feature extraction, and evaluation. Ensure that the dataset and classification type are chosen at the start.
