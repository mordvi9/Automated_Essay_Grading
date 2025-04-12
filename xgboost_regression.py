import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import numpy as np
import tensorflow as tf


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

def create_pipeline():
    """
    Create a pipeline for preprocessing and training the XGBRegressor model.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))  # XGBoost regression model
    ])
    return pipeline

def train_xgboost_regressor(X, y):
    """
    Perform hyperparameter tuning using a pipeline with XGBRegressor.
    """
    pipeline = create_pipeline()
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.8, 1.0]
    }
    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10, random_state=42, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    return model.predict(X)

def evaluate_model(model, X, y):
    """
    Evaluate the model using MSE, MAE, Huber Loss, Pearson Correlation, and QWK.
    """
    predictions = make_predictions(model, X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    huber = tf.keras.losses.Huber(delta=1.0)
    huber_loss = huber(y, predictions).numpy().mean()
    pearson_corr, _ = pearsonr(y, predictions)
    predictions_rounded = np.round(predictions).astype(int)
    y_test_rounded = np.round(y).astype(int)
    qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic')

    return mse, mae, huber_loss, pearson_corr, qwk, predictions

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data(r'./data/ielts_data.csv')
    data = preprocess_data(data)
    
    # Load extracted features from feature_extract.py
    features = preprocess_data(load_data(r'.\Extracted Features\extracted_features.csv'))
    
    # Feature engineering
    X = features.drop('score', axis=1)  # Replace 'score' with the actual target column name
    y = features['score']  # Replace 'score' with the actual target column name
    
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
        
        # Train the XGBoost regressor with hyperparameter tuning
        model = train_xgboost_regressor(X_train, y_train)
        
        # Evaluate the model
        mse, mae, huber_loss, pearson_corr, qwk, predictions = evaluate_model(model, X_test, y_test)
        mse_scores.append(mse)
        mae_scores.append(mae)
        huber_scores.append(huber_loss)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)

    # Calculate the average metrics across all folds
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