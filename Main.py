import Load_and_Preprocessing_Data as lp
import Model_Training as MT
import ML_Classifiers_Training_Validation as mlc
import Visualization as Vs
import Baseline_models as Bm
import Time_Frequency_Domain as TFD

# Define the datasets
BONN = 'bonn'
CHB_MIT = 'chbmit'
TUSZ = 'tusz'

# Ask the user to choose a dataset
dataset_choice = input("Choose a dataset (bonn, chbmit, tusz): ").lower()
classification_type = input("Choose a classification type (binary/multi): ").lower()

# Check the user's choice and set the appropriate dataset
if dataset_choice == BONN and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_bonn_data()
    history = MT.binary_model(eeg_data, eeg_label)
    Vs.Visualization_plots(history)
    mlc.binary_ML_Classifiers(eeg_data, eeg_label,dataset_choice)
    Bm.baseline_methods(eeg_data, eeg_label)
    TFD.TF_Analysis(frequency_domain=173.61)


elif dataset_choice == BONN and classification_type == 'multi':
    eeg_data, eeg_label = lp.load_bonn_data()
    history = MT.multi_class(eeg_data,eeg_label,dataset_choice,n_classes=5)
    Vs.Visualization_plots(history)
    mlc.Multi_ML_Classifiers(eeg_data,eeg_label,dataset_choice,n_classes=5)
    

elif dataset_choice == CHB_MIT and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_chbmit_data()
    history = MT.binary_model(eeg_data, eeg_label)
    Vs.Visualization_plots(history)
    mlc.binary_ML_Classifiers(eeg_data, eeg_label,dataset_choice)
    Bm.baseline_methods(eeg_data, eeg_label)
    TFD.TF_Analysis(frequency_domain=256)


elif dataset_choice == TUSZ and classification_type == 'binary':
    eeg_data, eeg_label = lp.load_tusz_data()
    history = MT.binary_model(eeg_data, eeg_label)
    Vs.Visualization_plots(history)
    mlc.binary_ML_Classifiers(eeg_data, eeg_label,dataset_choice)
    Bm.baseline_methods(eeg_data, eeg_label)
    TFD.TF_Analysis(frequency_domain=100)


elif dataset_choice == TUSZ and classification_type == 'multi':
    eeg_data, eeg_label = lp.load_tusz_data()
    history = MT.multi_class(eeg_data,eeg_label,dataset_choice,n_classes=8)
    Vs.Visualization_plots(history)
    mlc.Multi_ML_Classifiers(eeg_data,eeg_label,dataset_choice,n_classes=8)


else:
    print("Invalid dataset choice. Please choose either 'bonn', 'chbmit', or 'tusz'.")