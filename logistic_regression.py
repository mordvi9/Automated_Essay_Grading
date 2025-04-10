import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


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

def create_pipeline():
    """
    Create a pipeline for preprocessing and training the logistic regression model.
    """
    #TODO include the NLP features inside the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('classifier', LogisticRegression())  # Logistic regression model
    ])
    return pipeline

def train_logistic_regression(X, y):
    """
    Perform hyperparameter tuning using a pipeline.
    """
    pipeline = create_pipeline()
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga']
    }
    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10, random_state=42)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    return model.predict(X)

def evaluate_model(model, X, y):
    """
    Evaluate the model using Mean Squared Error (MSE).
    """
    predictions = make_predictions(model, X)
    mse = mean_squared_error(y, predictions)
    return mse, predictions

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('./data/ielts_data.csv')
    data = preprocess_data(data)
    
    #TODO Feature engineering
    X = data[['feature1', 'feature2', 'feature3']]  # Replace with actual feature columns
    y = data['target']  # Replace with actual target column
    
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    huber_scores = []
    pearson_scores = []
    qwk_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        
        # Train the logistic regression model with hyperparameter tuning
        model = train_logistic_regression(X_train, y_train)
        
         # Evaluate the model
        mse, mae, huber_loss, pearson_corr, qwk, predictions = model.evaluate(X_test, y_test)
        mse_scores.append(mse)
        mae_scores.append(mae)
        huber_scores.append(huber_loss)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)

    # Calculate the average MSE and MAE across all folds
    average_mse = np.mean(mse_scores)
    average_mae = np.mean(mae_scores)
    average_huber = np.mean(huber_scores)
    average_pearson = np.mean(pearson_scores)
    average_qwk = np.mean(qwk_scores)

    print(f'Average Mean Squared Error (5-Fold CV): {average_mse}')
    print(f'Average Mean Absolute Error (5-Fold CV): {average_mae}')
    print(f'Average Huber Loss (5-Fold CV): {average_huber}')
    print(f'Average Pearson Correlation (5-Fold CV): {average_pearson}')
    print(f'Average QWK (5-Fold CV): {average_qwk}')
