import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import numpy as np
import time  # For timing
import logging  # For debugging logs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SupportVectorRegressionModel:
    def __init__(self):
        """
        Initialize the SVR model with a pipeline for scaling.
        """
        self.model = make_pipeline(StandardScaler(), SVR())
    
    def train(self, X_train, y_train):
        """
        Train the SVR model with hyperparameter tuning.
        """
        logging.info("Starting hyperparameter tuning...")
        start_time = time.time()
        
        param_grid = {
            'svr__kernel': ['linear', 'rbf'],
            'svr__C': [1, 10, 100],
            'svr__epsilon': [0.1, 0.5]
        }
        grid_search = RandomizedSearchCV(
    self.model, param_grid, cv=5, scoring='neg_mean_squared_error', 
    n_iter=10, random_state=42, n_jobs=-1  # Use all CPU cores
)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        end_time = time.time()
        logging.info(f"Hyperparameter tuning completed in {end_time - start_time:.2f} seconds.")
    
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
        
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        pearson_corr, _ = pearsonr(y_test, predictions)
        predictions_rounded = np.round(predictions).astype(int)
        y_test_rounded = np.round(y_test).astype(int)
        qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic')
        
        end_time = time.time()
        logging.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
        return mse, mae, pearson_corr, qwk

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    logging.info(f"Loading data from {filepath}...")
    start_time = time.time()
    data = pd.read_csv(filepath)
    end_time = time.time()
    logging.info(f"Data loaded in {end_time - start_time:.2f} seconds.")
    return data

def preprocess_data(data):
    """
    Preprocess the data by handling missing values.
    """
    logging.info("Preprocessing data...")
    start_time = time.time()
    data = data.dropna()
    end_time = time.time()
    logging.info(f"Data preprocessing completed in {end_time - start_time:.2f} seconds.")
    return data

def main():
    logging.info("Starting main function...")
    start_time = time.time()
    
    # Load and preprocess data
    data = preprocess_data(load_data(r'./data/ielts_data.csv'))
    features = preprocess_data(load_data(r'./Extracted Features/extracted_features.csv'))
    
    # Merge data and features
    '''logging.info("Merging data and features...")
    merge_start_time = time.time()
    merged_df = pd.merge(data, features, left_index=True, right_index=True)
    merge_end_time = time.time()
    logging.info(f"Data and features merged in {merge_end_time - merge_start_time:.2f} seconds.")'''
    
    X = features.drop('score', axis=1)
    y = features['score']

    # Initialize the model
    svr_model = SupportVectorRegressionModel()

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores, mae_scores, pearson_scores, qwk_scores = [], [], [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        logging.info(f"Starting fold {fold}...")
        fold_start_time = time.time()
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train and evaluate the model
        svr_model.train(X_train, y_train)
        mse, mae, pearson_corr, qwk = svr_model.evaluate(X_test, y_test)
        mse_scores.append(mse)
        mae_scores.append(mae)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)
        
        fold_end_time = time.time()
        logging.info(f"Fold {fold} completed in {fold_end_time - fold_start_time:.2f} seconds.")

    # Calculate and print average metrics
    print(f'Average Mean Squared Error (5-Fold CV): {np.mean(mse_scores)}')
    print(f'Average Mean Absolute Error (5-Fold CV): {np.mean(mae_scores)}')
    print(f'Average Pearson Correlation (5-Fold CV): {np.mean(pearson_scores)}')
    print(f'Average QWK (5-Fold CV): {np.mean(qwk_scores)}')

    end_time = time.time()

if __name__ == "__main__":
    main()