import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, precision_score, recall_score
from scipy.stats import pearsonr
import numpy as np
import time  # For timing
import logging  # For debugging logs
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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

class SupportVectorRegressionModel:
    def __init__(self):
        """
        Initialize the SVR model with a pipeline for scaling.
        """
        self.model = None  # The model will be set after Optuna optimization

    def objective(self, trial, X_train, y_train, X_test, y_test):
        """
        Objective function for Optuna to optimize SVR hyperparameters.
        """
        # Define the hyperparameter search space
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        C = trial.suggest_float('C', 1, 100, log=True)
        epsilon = trial.suggest_float('epsilon', 0.1, 1.0)

        # Create the SVR model with the suggested hyperparameters
        model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon))

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model using Root Mean Squared Error (RMSE)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Return the RMSE as the objective to minimize
        return rmse

    def train(self, X_train, y_train, X_test, y_test, n_trials=20):
        """
        Train the SVR model using Optuna for hyperparameter optimization.
        """
        logging.info("Starting Optuna hyperparameter optimization...")
        start_time = time.time()

        # Create an Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)

        # Get the best hyperparameters
        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")

        # Train the final model with the best hyperparameters
        self.model = make_pipeline(StandardScaler(), SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon']))
        self.model.fit(X_train, y_train)

        end_time = time.time()
        logging.info(f"Optuna optimization completed in {end_time - start_time:.2f} seconds.")

    def predict(self, X_test):
        """
        Make predictions using the trained model.
        """
        logging.info("Making predictions...")
        start_time = time.time()
        predictions = self.model.predict(X_test)
        end_time = time.time()
        logging.info(f"Predictions completed in {end_time - start_time:.2f} seconds.")
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        logging.info("Evaluating the model...")
        start_time = time.time()

        # Make predictions
        predictions = self.predict(X_test)

        # Round predictions and true values to the nearest 0.5
        predictions_rounded = np.round(predictions * 2) / 2
        y_test_rounded = np.round(y_test * 2) / 2

        # Map rounded values to discrete labels
        unique_values = np.unique(np.concatenate([y_test_rounded, predictions_rounded]))
        value_to_label = {value: idx for idx, value in enumerate(unique_values)}
        y_test_labels = np.array([value_to_label[val] for val in y_test_rounded])
        predictions_labels = np.array([value_to_label[val] for val in predictions_rounded])

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        pearson_corr, _ = pearsonr(y_test, predictions)
        qwk = cohen_kappa_score(y_test_labels, predictions_labels, weights='quadratic')

        # Calculate precision and recall
        precision = precision_score(y_test_labels, predictions_labels, average='weighted', zero_division=0)
        recall = recall_score(y_test_labels, predictions_labels, average='weighted', zero_division=0)

        end_time = time.time()
        logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
        return rmse, mae, pearson_corr, qwk, precision, recall


def main():
    logging.info("Starting main function...")
    start_time = time.time()

    # Load and preprocess data
    data = preprocess_data(load_data(r'./data/ielts_data.csv'))
    features = preprocess_data(load_data(r'./Extracted Features/extracted_features.csv'))

    # Scale the score column to be out of 100
    features['score'] = features['score'] * (100 / 12)

    X = features.drop('score', axis=1)
    y = features['score']

    # Initialize the model
    svr_model = SupportVectorRegressionModel()

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores, mae_scores, pearson_scores, qwk_scores, precision_scores, recall_scores = [], [], [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        logging.info(f"Starting fold {fold}...")
        fold_start_time = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train and evaluate the model using Optuna
        svr_model.train(X_train, y_train, X_test, y_test, n_trials=20)
        rmse, mae, pearson_corr, qwk, precision, recall = svr_model.evaluate(X_test, y_test)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)
        precision_scores.append(precision)
        recall_scores.append(recall)

        fold_end_time = time.time()
        logging.info(f"Fold {fold} completed in {fold_end_time - fold_start_time:.2f} seconds.")

    # Calculate and print average metrics
    print(f'Average Root Mean Squared Error (5-Fold CV): {np.mean(rmse_scores)}')
    print(f'Average Mean Absolute Error (5-Fold CV): {np.mean(mae_scores)}')
    print(f'Average Pearson Correlation (5-Fold CV): {np.mean(pearson_scores)}')
    print(f'Average QWK (5-Fold CV): {np.mean(qwk_scores)}')
    print(f'Average Precision (5-Fold CV): {np.mean(precision_scores)}')
    print(f'Average Recall (5-Fold CV): {np.mean(recall_scores)}')

    end_time = time.time()
    logging.info(f"Main function completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()