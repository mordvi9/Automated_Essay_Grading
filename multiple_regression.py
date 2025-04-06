from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
import pandas as pd


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
    Create a pipeline for preprocessing and training the linear regression model.
    """
    #TODO include the NLP features inside the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('regressor', LinearRegression())  # Linear regression model
    ])
    return pipeline

def hyperparameter_tuning_with_pipeline(X_train, y_train):
    """
    Perform hyperparameter tuning using a pipeline.
    """
    param_grid = {
        'regressor__fit_intercept': [True, False],
        'regressor__normalize': [True, False]
    }
    pipeline = create_pipeline()
    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def make_predictions(model, X):
    return model.predict(X)

def evaluate_model(model, X, y):
    predictions = make_predictions(model, X)
    mse = mean_squared_error(y, predictions)
    return mse

if __name__ == "__main__":
    # Load and preprocess the data
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
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Train the model using hyperparameter tuning with the pipeline
        best_model = hyperparameter_tuning_with_pipeline(X_train, y_train)
        predictions = make_predictions(best_model, X_test)
        
        # Evaluate the model on the testing set
        mse = evaluate_model(y_test, predictions)
        mse_scores.append(mse)
        
        # Calculate MAE for this fold
        mae = np.mean(np.abs(y_test - predictions))
        mae_scores.append(mae)

    # Calculate the average MSE and MAE across all folds
    average_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    print(f'Average Mean Squared Error (5-Fold CV): {average_mse}')
    print(f'Average Mean Absolute Error (5-Fold CV): {avg_mae}')
    
    #TODO QWK