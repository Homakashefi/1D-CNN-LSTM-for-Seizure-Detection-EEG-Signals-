import os
import numpy as np
import random
import shutil
import zipfile
import mne
import wfdb
import pandas as pd
import pyedflib
import time
import logging
from glob import glob
import pickle
import glob
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import gc
from scipy.signal import find_peaks
import re
import tqdm

# 1. BONN Dataset
def load_bonn_data():
    # Unzip Five Different Classes of Bonn Epilepsy Dataset
    !unzip "/content/sample_data/z.zip" -d "/content/sample_data/SetA"
    !unzip "/content/sample_data/o.zip" -d "/content/sample_data/SetB"
    !unzip "/content/sample_data/n.zip" -d "/content/sample_data/SetC"
    !unzip "/content/sample_data/f.zip" -d "/content/sample_data/SetD"
    !unzip "/content/sample_data/s.zip" -d "/content/sample_data/SetE"


    # Defining Five Directories for Different Classes
    DATA_DIR_A = '/content/sample_data/SetA/Z/'
    DATA_DIR_B = '/content/sample_data/SetB/O/'
    DATA_DIR_C = '/content/sample_data/SetC/N/'
    DATA_DIR_D = '/content/sample_data/SetD/F/'
    DATA_DIR_E = '/content/sample_data/SetE/S/'

    # Defining labels for different classes
    LABEL_C1 = 0
    LABEL_C2 = 1
    LABEL_C3 = 2
    LABEL_C4 = 3
    LABEL_C5 = 4

    # Load Files From Folders
    def load_data():
        data = []
        label = []
        N = 0
        # Class A
        for fname in os.listdir(DATA_DIR_A):
            a = np.loadtxt(DATA_DIR_A + fname)
            data.append(a)
            label.append(np.array(LABEL_C1))
            N += 1

        # Class B
        for fname in os.listdir(DATA_DIR_B):
            b = np.loadtxt(DATA_DIR_B + fname)
            data.append(b)
            label.append(np.array(LABEL_C2))
            N += 1

        # Class C
        for fname in os.listdir(DATA_DIR_C):
            c = np.loadtxt(DATA_DIR_C + fname)
            data.append(c)
            label.append(np.array(LABEL_C3))
            N += 1

        # Class D
        for fname in os.listdir(DATA_DIR_D):
            d = np.loadtxt(DATA_DIR_D + fname)
            data.append(d)
            label.append(np.array(LABEL_C4))
            N += 1

        # Class E
        for fname in os.listdir(DATA_DIR_E):
            e = np.loadtxt(DATA_DIR_E + fname)
            data.append(e)
            label.append(np.array(LABEL_C2))
            N += 1

        return data, label

    data, label = load_data()
    return data, label

# 2. CHB-MIT Dataset
def load_chbmit_data():
    !wget -r -N -c -np physionet.org/files/chbmit/1.0.0/

    ch_labels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3','P3-O1',
                 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                 'FZ-CZ', 'CZ-PZ']

    path2pt = '/content/physionet.org/files/chbmit/1.0.0'
    folders = sorted(glob.glob(path2pt+'/*/'))
    n_patient = [m[-2:] for m in [l.rsplit('/', 2)[-2] for l in folders]]
    
    print(*n_patient)
    random.seed(2023)

    ratio_train = 0.99
    train_patient_str = sorted(random.sample(n_patient, round(ratio_train*len(n_patient))))
    test_patient_str = sorted([l for l in n_patient if l not in train_patient_str])
    print('Train PT: ', *train_patient_str)
    print('Test PT: ', *test_patient_str)

    files_train = []
    for l in train_patient_str:
        files_train = files_train + glob.glob(path2pt+'/chb{}/*.edf'.format(l))

    files_test = []
    for l in test_patient_str:
        files_test = files_test + glob.glob(path2pt+'/chb{}/*.edf'.format(l))
    
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler('read_files.log')
    logger.addHandler(fh)

    time_window = 8
    time_step = 4

    if os.path.exists('/kaggle/input/mit-chb-processed/signal_samples.npy') & os.path.exists('/kaggle/input/mit-chb-processed/is_sz.npy'):
        array_signals = np.load('/kaggle/input/mit-chb-processed/signal_samples.npy')
        array_is_sz = np.load('/kaggle/input/mit-chb-processed/is_sz.npy')
    else:
        p = 0.01
        counter = 0
        for temp_f in files_train:
            temp_edf = mne.io.read_raw_edf(temp_f)
            temp_labels = temp_edf.ch_names
            if sum([any([0 if re.match(c, l) == None else 1 for l in temp_edf.ch_names]) for c in ch_labels]) == len(ch_labels):
                time_window = 8
                time_step = 4
                fs = int(1 / (temp_edf.times[1] - temp_edf.times[0]))
                step_window = time_window * fs
                step = time_step * fs

                temp_is_sz = np.zeros((temp_edf.n_times,))
                if os.path.exists(temp_f + '.seizures'):
                    temp_annotation = wfdb.rdann(temp_f, 'seizures')
                    for i in range(int(temp_annotation.sample.size / 2)):
                        temp_is_sz[temp_annotation.sample[i * 2]:temp_annotation.sample[i * 2 + 1]] = 1
                temp_len = temp_edf.n_times

                temp_is_sz_ind = np.array(
                    [temp_is_sz[i * step:i * step + step_window].sum() / step_window for i in range((temp_len - step_window) // step)]
                )

                temp_0_sample_size = round(p * np.where(temp_is_sz_ind == 0)[0].size)
                temp_1_sample_size = np.where(temp_is_sz_ind > 0)[0].size

                counter = counter + temp_0_sample_size + temp_1_sample_size
            temp_edf.close()

        array_signals = np.zeros((counter, len(ch_labels), step_window), dtype=np.float32)
        array_is_sz = np.zeros(counter, dtype=bool)

        counter = 0
        for n, temp_f in enumerate(tqdm.tqdm(files_train)):
            to_log = 'No. {}: Reading. '.format(n)
            temp_edf = mne.io.read_raw_edf(temp_f)
            temp_labels = temp_edf.ch_names
            n_label_match = sum([any([0 if re.match(c, l) == None else 1 for l in temp_edf.ch_names]) for c in ch_labels])
            if n_label_match == len(ch_labels):
                ch_mapping = {sorted([l for l in temp_edf.ch_names if re.match(c, l) != None])[0]: c for c in ch_labels}
                temp_edf.rename_channels(ch_mapping)

                temp_is_sz = np.zeros((temp_edf.n_times,))
                temp_signals = temp_edf.get_data(picks=ch_labels) * 1e6

                if os.path.exists(temp_f + '.seizures'):
                    to_log = to_log + 'sz exists.'
                    temp_annotation = wfdb.rdann(temp_f, 'seizures')
                    for i in range(int(temp_annotation.sample.size / 2)):
                        temp_is_sz[temp_annotation.sample[i * 2]:temp_annotation.sample[i * 2 + 1]] = 1
                else:
                    to_log = to_log + 'No sz.'

                temp_len = temp_edf.n_times

                time_window = 8
                time_step = 4
                fs = int(1 / (temp_edf.times[1] - temp_edf.times[0]))
                step_window = time_window * fs
                step = time_step * fs

                temp_is_sz_ind = np.array(
                    [temp_is_sz[i * step:i * step + step_window].sum() / step_window for i in range((temp_len - step_window) // step)]
                )
                del temp_is_sz

                temp_0_sample_size = round(p * np.where(temp_is_sz_ind == 0)[0].size)
                temp_1_sample_size = np.where(temp_is_sz_ind > 0)[0].size

                # sz data
                temp_ind = list(np.where(temp_is_sz_ind > 0)[0])
                for i in temp_ind:
                    array_signals[counter, :, :] = temp_signals[:, i * step:i * step + step_window]
                    array_is_sz[counter] = True
                    counter = counter + 1

                # no sz data
                temp_ind = random.sample(list(np.where(temp_is_sz_ind == 0)[0]), temp_0_sample_size)
                for i in temp_ind:
                    array_signals[counter, :, :] = temp_signals[:, i * step:i * step + step_window]
                    array_is_sz[counter] = False
                    counter = counter + 1

                to_log += '{} signals added: {} w/o sz, {} w/ sz.'.format(
                    temp_0_sample_size + temp_1_sample_size, temp_0_sample_size, temp_1_sample_size
                )

            else:
                to_log += 'Not appropriate channel labels. Reading skipped.'.format(n)

            logger.info(to_log)
            temp_edf.close()

            if n % 10 == 0:
                gc.collect()
        gc.collect()

        np.save('signal_samples', array_signals)
        np.save('is_sz', array_is_sz)

    array_signals = array_signals[:, :, ::2]

    array_n = np.where(array_is_sz > .0)[0]
    print('Number of all the extracted signals: {}'.format(array_is_sz.size))
    print('Number of signals with seizures: {}'.format(array_n.size))
    print('Ratio of signals with seizures: {:.3f}'.format(array_n.size / array_is_sz.size))
    array_signals = array_signals[:, :, :, np.newaxis]

    from sklearn import model_selection
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        array_signals, array_is_sz, test_size=0.0,
        stratify=(array_is_sz > 0))

    del array_signals, array_is_sz
    return X_train, y_train

# 3. TUSZ Dataset
def load_tusz_data():
    from google.colab import drive
    drive.mount('/content/drive')

    # Extract zip files for TUSZ dataset
    zip_files = glob.glob('/content/drive/MyDrive/aaaaa*.zip')

    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/content')
            print(f"Extracted: {zip_path}")

    source_pattern = '/content/aaaaa*'
    destination = '/content/test'

    # Ensure destination exists
    os.makedirs(destination, exist_ok=True)

    # Find all folders matching the pattern
    matching_dirs = [d for d in glob(source_pattern) if os.path.isdir(d)]

    # Move each matching folder into the destination
    for dir_path in matching_dirs:
        folder_name = os.path.basename(dir_path)
        destination_path = os.path.join(destination, folder_name)

        # Avoid overwriting existing folders
        if os.path.exists(destination_path):
            print(f"Skipping {dir_path} — folder already exists in destination.")
            continue

        shutil.move(dir_path, destination)

    # Define the base directory of your dataset

    base_dir = "/content/test/aaaaa***"
    SEIZURE_CLASSES = ['fnsz', 'gnsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'spsz', 'mysz']
    
    # Function to run shell commands and capture the output
    def run_command(cmd):
        return subprocess.getoutput(cmd)

    # (11) Total number of seizure events per seizure type:
    print("No. of Seizure Events")
    for seizure_type in SEIZURE_CLASSES:
        cmd = f"find {base_dir} -name '*.csv' -exec grep -H {seizure_type} {{}} \; | wc -l"
        print(f"{seizure_type}: {run_command(cmd)} ")

    # (12) Total duration of seizure events per seizure type:
    print("Duration(s)")
    for seizure_type in SEIZURE_CLASSES:
        cmd = f"find {base_dir} -name '*.csv' -exec grep -H '{seizure_type},' {{}} \; | cut -d',' -f2,3 | sed -e 's/,/ /g' | awk '{{ sum +=($2-$1)}} END {{print sum}}'"
        print(f"{seizure_type}: {run_command(cmd)} ")

    # (13) Number of patients with specific seizure type:
    print("No. of Patients")
    for seizure_type in SEIZURE_CLASSES:
        cmd = f"find {base_dir} -name '*.csv' -exec grep -H '{seizure_type},' {{}} \; | cut -d'/' -f8 | sort -u | wc -l"
        print(f"{seizure_type}: {run_command(cmd)} ")
    
    # Directories and Event Classes
    TUSZ_DIR = "/content/test/aaaaa***"  # change this to match your folder structure
    EVENT_CLASSES = ['fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz']

    # Define a function to print out the contents of an EDF file
    def print_edf_contents(edf_file_path):
        try:
            f = pyedflib.EdfReader(edf_file_path)
            num_signals = f.signals_in_file
            signal_labels = f.getSignalLabels()
            print(f"Number of Signals: {num_signals}")
            print("Signal Labels:", signal_labels)
            for i in range(num_signals):
                print(f"Signal {i+1} ({signal_labels[i]}):")
                print(f.readSignal(i)[:10])  # Print first 10 values of each signal to show user
        finally:
            f.close()

    # Define a function to print out the contents of a CSV file
    def print_csv_contents(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path, sep=",", comment='#', skip_blank_lines=True)
            print("\nCSV File Contents:")
            print(df.head(10))  # Print the first 5 rows to show user
        except pd.errors.ParserError as e:
            print(f"ParserError for CSV file {csv_file_path}: {e}")
        except Exception as e:
            print(f"Error reading CSV file {csv_file_path}: {e}")

    # Example usage
    print_edf_contents("/content/test/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf")
    print_csv_contents("/content/test/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.csv")
    # Mapping from standard montage to EDF file channels for both -REF and -LE
    MONTAGE_MAP_REF = {
        'FP1-F7': ('EEG FP1-REF', 'EEG F7-REF'),
        'F7-T3': ('EEG F7-REF', 'EEG T3-REF'),
        'T3-T5': ('EEG T3-REF', 'EEG T5-REF'),
        'T5-O1': ('EEG T5-REF', 'EEG O1-REF'),
        'FP2-F8': ('EEG FP2-REF', 'EEG F8-REF'),
        'F8-T4': ('EEG F8-REF', 'EEG T4-REF'),
        'T4-T6': ('EEG T4-REF', 'EEG T6-REF'),
        'T6-O2': ('EEG T6-REF', 'EEG O2-REF'),
        'A1-T3': ('EEG A1-REF', 'EEG T3-REF'),
        'T3-C3': ('EEG T3-REF', 'EEG C3-REF'),
        'C3-CZ': ('EEG C3-REF', 'EEG CZ-REF'),
        'CZ-C4': ('EEG CZ-REF', 'EEG C4-REF'),
        'C4-T4': ('EEG C4-REF', 'EEG T4-REF'),
        'T4-A2': ('EEG T4-REF', 'EEG A2-REF'),
        'FP1-F3': ('EEG FP1-REF', 'EEG F3-REF'),
        'F3-C3': ('EEG F3-REF', 'EEG C3-REF'),
        'C3-P3': ('EEG C3-REF', 'EEG P3-REF'),
        'P3-O1': ('EEG P3-REF', 'EEG O1-REF'),
        'FP2-F4': ('EEG FP2-REF', 'EEG F4-REF'),
        'F4-C4': ('EEG F4-REF', 'EEG C4-REF'),
        'C4-P4': ('EEG C4-REF', 'EEG P4-REF'),
        'P4-O2': ('EEG P4-REF', 'EEG O2-REF')
    }

    MONTAGE_MAP_LE = {
        'FP1-F7': ('EEG FP1-LE', 'EEG F7-LE'),
        'F7-T3': ('EEG F7-LE', 'EEG T3-LE'),
        'T3-T5': ('EEG T3-LE', 'EEG T5-LE'),
        'T5-O1': ('EEG T5-LE', 'EEG O1-LE'),
        'FP2-F8': ('EEG FP2-LE', 'EEG F8-LE'),
        'F8-T4': ('EEG F8-LE', 'EEG T4-LE'),
        'T4-T6': ('EEG T4-LE', 'EEG T6-LE'),
        'T6-O2': ('EEG T6-LE', 'EEG O2-LE'),
        'A1-T3': ('EEG A1-LE', 'EEG T3-LE'),
        'T3-C3': ('EEG T3-LE', 'EEG C3-LE'),
        'C3-CZ': ('EEG C3-LE', 'EEG CZ-LE'),
        'CZ-C4': ('EEG CZ-LE', 'EEG C4-LE'),
        'C4-T4': ('EEG C4-LE', 'EEG T4-LE'),
        'T4-A2': ('EEG T4-LE', 'EEG A2-LE'),
        'FP1-F3': ('EEG FP1-LE', 'EEG F3-LE'),
        'F3-C3': ('EEG F3-LE', 'EEG C3-LE'),
        'C3-P3': ('EEG C3-LE', 'EEG P3-LE'),
        'P3-O1': ('EEG P3-LE', 'EEG O1-LE'),
        'FP2-F4': ('EEG FP2-LE', 'EEG F4-LE'),
        'F4-C4': ('EEG F4-LE', 'EEG C4-LE'),
        'C4-P4': ('EEG C4-LE', 'EEG P4-LE'),
        'P4-O2': ('EEG P4-LE', 'EEG O2-LE')
    }
    def parse_csv_annotations(csv_file):
    
    #Extracts seizure event annotations from a .csv file.
    
        try:
            df = pd.read_csv(csv_file, sep=",", comment='#', skip_blank_lines=True)
            df.columns = ['channel', 'start_time', 'stop_time', 'label', 'confidence']
            df = df[['start_time', 'stop_time', 'channel', 'label']]  # Keep relevant columns
            return df[df['label'].isin(EVENT_CLASSES)]
        except pd.errors.ParserError as e:
            print(f"ParserError for file {csv_file}: {e}")
            return pd.DataFrame()
        except ValueError as ve:
            print(f"ValueError for file {csv_file}: {ve}")
            return pd.DataFrame()
    
    def get_signal_difference(f, ch1, ch2, start, stop):

        #Computes the difference between two EEG channels.
        signal1 = f.readSignal(ch1)[start:stop]
        signal2 = f.readSignal(ch2)[start:stop]
        return signal1 - signal2

    def load_eeg_with_annotations(edf_file, annotations, montage_map, num_timesteps=256):
        """
        Loads EEG data corresponding to annotated events.
        """
        segments = []
        try:
            start_time = time.time()
            f = pyedflib.EdfReader(edf_file)
            channel_labels = f.getSignalLabels()
            sample_freq = f.getSampleFrequency(0)
            print("Channels:", channel_labels)
            print("Sample Frequency:", sample_freq)
            print("EDF Reading Time:", time.time() - start_time)

            for _, row in annotations.iterrows():
                start = int(row['start_time'] * sample_freq)
                stop = int(row['stop_time'] * sample_freq)
                signal = np.zeros((len(montage_map), num_timesteps))

                annotation_start_time = time.time()
                for idx, (montage, (ch1, ch2)) in enumerate(montage_map.items()):
                    if ch1 in channel_labels and ch2 in channel_labels:
                        ch1_index = channel_labels.index(ch1)
                        ch2_index = channel_labels.index(ch2)
                        channel_signal = get_signal_difference(f, ch1_index, ch2_index, start, stop)

                        # Ensure the channel_signal is exactly num_timesteps long
                        if len(channel_signal) < num_timesteps:
                            # Pad the signal if it is shorter
                            padded_signal = np.pad(channel_signal, (0, num_timesteps - len(channel_signal)), 'constant')
                            signal[idx, :] = padded_signal
                        else:
                            # Truncate the signal if it is longer
                            signal[idx, :] = channel_signal[:num_timesteps]
                    else:
                        print(f"Channels {ch1} or {ch2} not found in EDF file.")
                print("Annotation Processing Time:", time.time() - annotation_start_time)

                segments.append((signal, row['label']))
        except Exception as e:
            print(f"Error loading EDF file {edf_file}: {e}")
        finally:
            if 'f' in locals():
                f.close()

        return segments

    def preprocess_segments(segments):
        """
        Preprocess EEG segments: normalize and pad/truncate.
        """
        preprocessed_segments = []
        for signal, event in segments:
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            normalized_signal = (signal - mean) / std
            preprocessed_segments.append((normalized_signal, event))
        return preprocessed_segments
    
    def save_checkpoint(data, filename):
        """
        Save checkpoint data to a file atomically.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Use a temporary filename for atomic save
        temp_filename = filename + '.tmp'
        with open(temp_filename, 'wb') as f:
            pickle.dump(data, f)
        os.rename(temp_filename, filename)

    def load_checkpoint(filename):
        """
        Load checkpoint data from a file with error handling.
        """
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Error loading checkpoint {filename}: {e}")
                return None
        return None
    
    def load_and_preprocess_all_segments(directory, num_timesteps=256, checkpoint_dir='checkpoints'):
        """
        Load and preprocess all EEG segments from all subdirectories with checkpointing.
        """
        all_segments = []
        processed_patients = set()

        # Load checkpoint if exists
        checkpoint_file = os.path.join(checkpoint_dir, 'all_segments.pkl')
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data is not None:
            all_segments, processed_patients = checkpoint_data

        for root, _, files in os.walk(directory):
            edf_files = [f for f in files if f.endswith('.edf')]
            for edf_file in edf_files:
                base_name = edf_file[:-4]
                csv_file = os.path.join(root, base_name + '.csv')
                if os.path.exists(csv_file) and base_name not in processed_patients:
                    annotations = parse_csv_annotations(csv_file)
                    edf_path = os.path.join(root, edf_file)

                    # Determine montage type based on directory structure
                    if '02_tcp_le' in root:
                        montage_map = MONTAGE_MAP_LE
                    elif '03_tcp_ar_a' in root:
                        montage_map = {k: v for k, v in MONTAGE_MAP_REF.items() if 'EEG A1-REF' not in v and 'EEG A2-REF' not in v}
                    else:
                        montage_map = MONTAGE_MAP_REF

                    segments = load_eeg_with_annotations(edf_path, annotations, montage_map, num_timesteps)
                    preprocessed_segments = preprocess_segments(segments)
                    all_segments.extend(preprocessed_segments)
                    processed_patients.add(base_name)

                    # Save checkpoint after processing each patient
                    save_checkpoint((all_segments, processed_patients), checkpoint_file)

        print(f"Processed {len(processed_patients)} patients from directory {directory}")
        return all_segments
    
    # Ensure checkpoint directory exists
    os.makedirs('/content/DDData', exist_ok=True)

    def prepare_data(segments):
        """
        Prepares data and labels for deep learning.
        """
        X = []
        y = []
        label_map = {event: idx for idx, event in enumerate(EVENT_CLASSES)}
        for segment in segments:
            if len(segment) == 2:
                signal, event = segment
            else:
                print(f"Unexpected segment format: {segment}")
                continue

            # Check if the signal shape is consistent with expected shape (22 channels, 256 timesteps)
            if signal.shape != (22, 256):
                #print(f"Inconsistent signal shape detected: {signal.shape}")
                continue  # Skip this signal if the shape is inconsistent

            X.append(signal)
            y.append(label_map[event])

        return np.array(X), np.array(y)
    

    # Path to all folders matching the pattern
    input_folders = glob.glob('/content/test/aaaaa***')

    # Initialize containers for all data and labels
    all_data = []
    all_labels = []

    # Loop through each folder
    for folder in input_folders:
        print(f"Processing: {folder}")

        # Load and preprocess segments
        test_segments = load_and_preprocess_all_segments(folder, checkpoint_dir='/content/DDData')

        # Prepare data and labels
        data, label = prepare_data(test_segments)

        # Collect results
        all_data.append(data)
        all_labels.append(label)

        print("Shape:", data.shape, label.shape)

    # If you want to concatenate all data into one array (optional, depending on usage)
    import numpy as np
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("Final Combined Data Shape:", data.shape, labels.shape)


