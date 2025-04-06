import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

def build_model(input_shape, learning_rate=0.001):
    """
    This function builds a neural network model for regression tasks.
    """
    #TODO experiment with different activation functions and optimizers
    #TODO add regularization techniques
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))  # Output layer for regression
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

class NeuralNetworkPipeline:
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=32):
        """
        Initialize the pipeline with hyperparameters for the neural network.
        """
        #TODO add NLP features to the pipeline
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train, y_train):
        """
        Train the neural network model with scaled data.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = build_model(input_shape=(X_train_scaled.shape[1],), learning_rate=self.learning_rate)
        #TODO possibly add regularization (early stopping)
        self.model.fit(X_train_scaled, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

    def predict(self, X_test):
        """
        Make predictions using the trained model.
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled).flatten()

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Mean Squared Error (MSE).
        """
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse, predictions

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Preprocess the data by handling missing values, encoding categorical variables, etc.
    """
    data = data.dropna()  # Drop rows with missing values
    return data

def main():
    #TODO explore more data
    data = load_data('ielts_writing_dataset.csv')
    data = preprocess_data(data)
    
    # Feature engineering
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']  # Replace 'target_column' with the actual target column name

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize and train the pipeline
        nn_pipeline = NeuralNetworkPipeline(learning_rate=0.001, epochs=100, batch_size=32)
        nn_pipeline.fit(X_train, y_train)

        # Evaluate the pipeline
        mse, predictions = nn_pipeline.evaluate(X_test, y_test)
        mse_scores.append(mse)

        # Calculate MAE for this fold
        mae = np.mean(np.abs(predictions - y_test))
        mae_scores.append(mae)

    # Calculate the average MSE and MAE across all folds
    average_mse = np.mean(mse_scores)
    average_mae = np.mean(mae_scores)
    print(f'Average Mean Squared Error (5-Fold CV): {average_mse}')
    print(f'Average Mean Absolute Error (5-Fold CV): {average_mae}')
    
    #TODO QWK

if __name__ == "__main__":
    main()