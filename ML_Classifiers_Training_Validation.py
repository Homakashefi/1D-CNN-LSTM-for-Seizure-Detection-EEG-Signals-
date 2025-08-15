
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pywt
import Extract_Wavelet_Coefficients as EX # pyright: ignore[reportMissingImports]
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras import regularizers
import keras
import shap
from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Conv1D, MaxPooling1D, Flatten
import pywt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
from itertools import cycle
import tensorflow as tf
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score, cohen_kappa_score, roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
import tensorflow

def plot_roc_per_fold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2 # this is already defined here
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) #, linewidth=20) remove this duplicate argument
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--') #, linewidth=20) remove this duplicate argument
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver operating characteristic', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    # Instead of fontdict, use prop keyword argument
    plt.legend(loc="lower right", prop={'size': 20, 'weight': 'bold'})
    plt.show()

def plot_multiclass_roc(y_true, y_prob):
    """
    Plots the ROC curves for multi-class classification.

    Args:
        y_true: True labels (one-hot encoded).
        y_prob: Predicted probabilities for each class.
    """
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (area = {roc_auc["micro"]:.2f})', linestyle="--", linewidth=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') #, linewidth=20) remove this duplicate argument
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel('True Positive Rate', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.title('Receiver operating characteristic', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    # Instead of fontdict, use prop keyword argument
    plt.legend(loc="lower right", prop={'size': 10, 'weight': 'bold'})
    plt.show()    

def binary_ML_Classifiers(data, label,dataset_choice):
    # Define classifiers
    classifiers = {
        'SVC': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'GB': GradientBoostingClassifier(),
        'RF': RandomForestClassifier(),
        'GNB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'MLP': MLPClassifier()
    }

    # Create a dictionary to store metrics for each classifier
    classifiers_metrics = {name: {'Accuracy': [], 'Precision': [], 'Specificity': [], 'Sensitivity': [], 'NPV': [], 'PPV': [], 'F1': [], 'MCC': [], 'Kappa':[], 'GDR':[]} for name in classifiers}

    kf = KFold(n_splits=10, shuffle=True, random_state=2)

    for train_index, test_index in kf.split(data):
        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]

        #Preparing dataset as an input for wavelet and further processing
        if dataset_choice == 'bonn':
            X_train = X_tr_va.reshape(X_tr_va.shape[0], 4097, 1)
            Y_train = y_tr_va.reshape(y_tr_va.shape[0], 1)


            X_test = X_test.reshape(X_test.shape[0], 4097, 1)
            Y_test = y_test.reshape(y_test.shape[0], 1)


            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train, level=3)
            features_Test = EX.extract_coeffs_BONN_CHB(X_test, level=3)

            #Expanding the dimensions to be prepared as an input to CNN-LSTM model

            EEG_Train = features_Train.reshape(features_Train.shape[0], 4100, 1)
            EEG_Test = features_Test.reshape(features_Test.shape[0], 4100, 1)

            # Reshape the input data for the SVC classifier
            EEG_Train_reshaped = EEG_Train.reshape(EEG_Train.shape[0], -1)
            EEG_Test_reshaped = EEG_Test.reshape(EEG_Test.shape[0], -1)

        if dataset_choice == 'chbmit':

            X_train, X_test = X_train[train_index], X_train[test_index]
            Y_train, y_test = Y_train[train_index], Y_train[test_index]

            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train, level=3)
            features_Test = EX.extract_coeffs_BONN_CHB(X_test, level=3)

            #Expanding the dimensions to be prepared as an input to CNN-LSTM model

            EEG_Train = features_Train.reshape(features_Train.shape[0],72, 1)
            EEG_Test = features_Test.reshape(features_Test.shape[0], 72, 1)
            
            # Reshape the input data for the SVC classifier
            EEG_Train_reshaped = EEG_Train.reshape(features_Train.shape[0], -1)
            EEG_Test_reshaped = EEG_Test.reshape(features_Test.shape[0], -1)

        if dataset_choice == 'tusz':
            X_train, X_test = data[train_index], data[test_index]
            Y_train, y_test = label[train_index], label[test_index]

            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_TUSZ(X_train, level=3)
            features_Test = EX.extract_coeffs_TUSZ(X_test, level=3)

            # Reshape the data to 2D for traditional classifiers
            EEG_Train = features_Train.reshape(features_Train.shape[0], -1)  # Reshape to (n_samples, n_features)
            EEG_Test = features_Test.reshape(features_Test.shape[0], -1)  # Reshape to (n_samples, n_features)

            # One-hot encoding using full label space
            Y_train = to_categorical(Y_train, num_classes=2)  
            y_test = to_categorical(y_test, num_classes=2)  


        for name, clf in classifiers.items():
            clf.fit(EEG_Train_reshaped, Y_train)
            y_pred = clf.predict(EEG_Test_reshaped)
            y_proba = clf.predict_proba(EEG_Test_reshaped)[:, 1]

            # Calculate confusion matrix
            # tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel() # This line caused the error
            cm = confusion_matrix(Y_test, y_pred)  # Get the confusion matrix
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else [0, 0, 0, 0] # Handle multi-class case

            # Calculate metrics
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Avoid division by zero
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Avoid division by zero
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Avoid division by zero
            ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Avoid division by zero
            f1 = f1_score(Y_test, y_pred)
            mcc = matthews_corrcoef(Y_test, y_pred)
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred)
            # Calculate Cohen's Kappa
            kappa = cohen_kappa_score(y_test, y_pred)
            # Calculate GDR (Geometric Mean of Sensitivity and Specificity)
            gdr = np.sqrt(sensitivity * specificity)


            # Append the metrics to the classifier's specific list in the dictionary
            classifiers_metrics[name]['Accuracy'].append(accuracy)
            classifiers_metrics[name]['Precision'].append(precision)
            classifiers_metrics[name]['Specificity'].append(specificity)
            classifiers_metrics[name]['Sensitivity'].append(sensitivity)
            classifiers_metrics[name]['NPV'].append(npv)
            classifiers_metrics[name]['PPV'].append(ppv)
            classifiers_metrics[name]['F1'].append(f1)
            classifiers_metrics[name]['MCC'].append(mcc)
            classifiers_metrics[name]['Kappa'].append(Kappa)
            classifiers_metrics[name]['GDR'].append(gdr)
            plot_roc_per_fold(Y_test, y_pred)



    # Calculate the mean of each metric for each classifier and print the results
    for name, metrics in classifiers_metrics.items():
        print(f"Average metrics for {name}:")
        for metric, scores in metrics.items():
            print(f"{metric}: {np.mean(scores)}")
        print("----------")


def Multi_ML_Classifiers(data,label,n_classes,dataset_choice):

    # Define classifiers
    classifiers = {
        'SVC': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'GB': GradientBoostingClassifier(),
        'RF': RandomForestClassifier(),
        'GNB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'MLP': MLPClassifier()
    }

    # Create a dictionary to store metrics for each classifier
    classifiers_metrics = {name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'Sensitivity': [], 'NPV': [], 'PPV': [], 'F1': [], 'MCC': [], 'Kappa':[], 'GDR':[]} for name in classifiers}

    kf = KFold(n_splits=10, shuffle=True, random_state=2)

    for train_index, test_index in kf.split(data):
        X_tr_va, X_test = data[train_index], data[test_index]
        y_tr_va, y_test = label[train_index], label[test_index]
            
        if dataset_choice == 'bonn':
            X_train = X_tr_va.reshape(X_tr_va.shape[0], 4097, 1)
            Y_train = y_tr_va.reshape(y_tr_va.shape[0], 1)


            X_test = X_test.reshape(X_test.shape[0], 4097, 1)
            Y_test = y_test.reshape(y_test.shape[0], 1)


            # Extract Wavelet coefficients from the EEG data (Test and Train)
            features_Train = EX.extract_coeffs_BONN_CHB(X_train, level=3)
            features_Test = EX.extract_coeffs_BONN_CHB(X_test, level=3)

            #Expanding the dimensions to be prepared as an input to CNN-LSTM model

            EEG_Train = features_Train.reshape(features_Train.shape[0], 4100, 1)
            EEG_Test = features_Test.reshape(features_Test.shape[0], 4100, 1)

            # Reshape the input data for the SVC classifier
            EEG_Train_reshaped = EEG_Train.reshape(EEG_Train.shape[0], -1)
            EEG_Test_reshaped = EEG_Test.reshape(EEG_Test.shape[0], -1)

        if dataset_choice == 'tusz':
            X_train, X_test = data[train_index], data[test_index]
            Y_train, y_test = label[train_index], label[test_index]

            # Wavelet transformation
            features_train = EX.extract_coeffs_TUSZ(X_train, level=3)
            features_test = EX.extract_coeffs_TUSZ(X_test, level=3)

            EEG_Train_reshaped = features_train.reshape(features_train.shape[0], -1)
            EEG_Test_reshaped = features_test.reshape(features_test.shape[0], -1)

        # Binarize the labels for multi-class metrics calculation
        Y_train_binarized = label_binarize(Y_train, classes=np.unique(Y_train))
        Y_test_binarized = label_binarize(Y_test, classes=np.unique(Y_test))
        Y_test_labels = np.argmax(Y_test, axis=1)

        for name, clf in classifiers.items():
            clf.fit(EEG_Train_reshaped, Y_train.ravel())
            y_pred = clf.predict(EEG_Test_reshaped)
            y_prob = clf.predict_proba(EEG_Test_reshaped)
            y_pred_binarized = label_binarize(y_pred, classes=np.unique(Y_train))

            # Calculate metrics
            accuracy = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred, average='weighted')
            recall = recall_score(Y_test, y_pred, average='weighted')
            f1 = f1_score(Y_test, y_pred, average='weighted')
            mcc = matthews_corrcoef(Y_test, y_pred)


            # Calculate specificity, sensitivity, NPV, and PPV
            cm = confusion_matrix(Y_test_binarized.argmax(axis=1), y_pred_binarized.argmax(axis=1))
            specificity = np.sum(np.diag(cm)) / np.sum(cm)
            sensitivity = recall  # Sensitivity is the same as recall in multi-class
            npv = np.sum(np.diag(cm)) / np.sum(cm.sum(axis=0) - np.diag(cm))
            ppv = precision  # PPV is the same as precision in multi-class
            # Calculate Cohen's Kappa
            kappa = cohen_kappa_score(y_test, y_pred)
            # Calculate GDR (Geometric Mean of Sensitivity and Specificity)
            gdr = np.sqrt(sensitivity * specificity)

            # Append the metrics to the classifier's specific list in the dictionary
            classifiers_metrics[name]['Accuracy'].append(accuracy)
            classifiers_metrics[name]['Precision'].append(precision)
            classifiers_metrics[name]['Recall'].append(recall)
            classifiers_metrics[name]['Specificity'].append(specificity)
            classifiers_metrics[name]['Sensitivity'].append(sensitivity)
            classifiers_metrics[name]['NPV'].append(npv)
            classifiers_metrics[name]['PPV'].append(ppv)
            classifiers_metrics[name]['F1'].append(f1)
            classifiers_metrics[name]['MCC'].append(mcc)
            classifiers_metrics[name]['Kappa'].append(Kappa)
            classifiers_metrics[name]['GDR'].append(gdr)
            plot_multiclass_roc(Y_test, y_prob)

    # Calculate the mean of each metric for each classifier and print the results
    for name, metrics in classifiers_metrics.items():
        print(f"Average metrics for {name}:")
        for metric, scores in metrics.items():
            print(f"{metric}: {np.mean(scores)}")
        print("----------")

