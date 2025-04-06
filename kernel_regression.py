from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class KernelRegressionModel:
    def __init__(self):
        """
        Initialize the pipeline with a StandardScaler and KernelRidge model.
        """
        #TODO add NLP features to the pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Standardize the features
            ('kernel_ridge', KernelRidge())  # Kernel Ridge Regression model
        ])
    
    def train(self, X, y):
        """
        Train the Kernel Ridge Regression model.
        """
        self.pipeline.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions using the trained pipeline.
        """
        return self.pipeline.predict(X)
    
    def tune_hyperparameters(self, X, y):
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        #TODO experiment with custom kernels possibly
        #TODO experiment with different regularization techniques
        param_grid = {
            'kernel_ridge__alpha': [0.1, 1.0, 10.0],
            'kernel_ridge__kernel': ['linear', 'polynomial', 'rbf']
        }
        grid_search = RandomizedSearchCV(self.pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        self.pipeline = grid_search.best_estimator_
        return grid_search.best_params_

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
    # Example preprocessing steps (customize as needed):
    data = data.dropna()  # Drop rows with missing values
    # Add more preprocessing steps here (e.g., encoding categorical variables)
    return data

def main():
    # Load and preprocess the dataset
    data = load_data('ielts_writing_dataset.csv')
    data = preprocess_data(data)
    
    # Feature engineering
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']  # Replace 'target_column' with the actual target column name

    # Initialize the model
    kernel_regression_model = KernelRegressionModel()

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Hyperparameter tuning and training
        kernel_regression_model.tune_hyperparameters(X_train, y_train)
        kernel_regression_model.train(X_train, y_train)
        
        # Make predictions and evaluate
        predictions = kernel_regression_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)
        
        # Calculate MAE for this fold
        mae = np.mean(np.abs(predictions - y_test))
        mae_scores.append(mae)

    # Calculate the average MSE and MAE across all folds
    average_mse = np.mean(mse_scores)
    average_mae = np.mean(mae_scores)
    print(f'Average Mean Squared Error (5-Fold CV): {average_mse}')
    print(f'Average Mean Absolute Error (5-Fold CV): {average_mae}')
    
    # Calculate Quadratic Weighted Kappa
    # TODO Implement QWK calculation here

if __name__ == "__main__":
    main()