import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def baseline_methods(data, label):
    # Feature scaling
    scaler = StandardScaler()
    data = scaler.fit_transform(data)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)


    # Define and train models

    # CNN
    def create_cnn_model(input_shape):
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Reshape data for CNN (assuming 1 channel)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    cnn_model = create_cnn_model((X_train_cnn.shape[1], 1))
    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.1)
    cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

    # SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)


    # MLP
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)


    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)

    # Gaussian Naive Bayes
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)
    gnb_pred = gnb_model.predict(X_test)


    # Evaluate models
    print("CNN Accuracy:", accuracy_score(y_test, cnn_pred))
    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
    print("MLP Accuracy:", accuracy_score(y_test, mlp_pred))
    print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
    print("GNB Accuracy:", accuracy_score(y_test, gnb_pred))