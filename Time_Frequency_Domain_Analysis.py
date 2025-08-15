import numpy as np
from scipy import signal #signal is a package, not a variable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def TF_Analysis(data, label,frequency_domain):
    # Feature extraction (Time-domain and Frequency-domain)
    def extract_features():
        features = []
        for signal_data in data: #changed 'signal' to 'signal_data'
            # Time-domain features
            mean = np.mean(signal_data)
            std_dev = np.std(signal_data)
            rms = np.sqrt(np.mean(signal_data**2))

            # Frequency-domain features
            frequencies, psd = signal.welch(signal_data, fs= frequency_domain)  # Assuming sampling rate of 173.61 Hz (adjust if different)
            mean_freq = np.mean(frequencies)
            median_freq = np.median(frequencies)

            features.append([mean_freq, median_freq])
        return np.array(features)

    # Prepare the data
    features = extract_features(data)

    # Split into training and testing sets
    XX_train, XX_test, yy_train, yy_test = train_test_split(features, label, test_size=0.20, random_state=42)

    # Train a classifier (e.g., RandomForest)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(XX_train, yy_train)

    # Make predictions
    y_pred = clf.predict(XX_test)

    # Calculate accuracy
    accuracy = accuracy_score(yy_test, y_pred)
    print(f"Accuracy: {accuracy}")