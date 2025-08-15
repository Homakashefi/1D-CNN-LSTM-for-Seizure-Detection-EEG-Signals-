import PyWavelets  # type: ignore
import numpy as np
import pywt


#Extracting approximation and Detailed Coefficients from signal

def extract_coeffs_BONN_CHB(eeg_data, level=3):
    coeffs_list = []

    for i in range(eeg_data.shape[0]):
        # Select the EEG signal from the ith trial
        signal = eeg_data[i, :, 0]

        # Apply DWT on the signal
        coeffs = pywt.wavedec(signal, 'db1', level=level)  # 'db1' refers to Daubechies wavelet

        # Flatten the coefficients and append to the list
        coeffs_flattened = np.concatenate(coeffs)
        coeffs_list.append(coeffs_flattened)

    return np.array(coeffs_list)

def extract_coeffs_TUSZ(eeg_data, level=3, wavelet='db1'):

    n_samples, n_channels, timepoints, _ = eeg_data.shape
    coeffs_list = []

    for i in range(n_samples):
        sample_coeffs = []
        for ch in range(n_channels):
            signal = eeg_data[i, ch, :, 0]  # shape (256,)
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            coeffs_flat = np.concatenate(coeffs)
            sample_coeffs.append(coeffs_flat)
        coeffs_list.append(sample_coeffs)

    return np.array(coeffs_list)  # shape: (n_samples, n_channels, coeff_length)

