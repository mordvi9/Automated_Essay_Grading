import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
import numpy as np
import optuna
import argparse
import csv


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


def create_pipeline(params, n_features):
    """
    Create a pipeline for preprocessing and feature selection.
    """
    regressor = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],  # L1 regularization
        reg_lambda=params['reg_lambda']  # L2 regularization
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize the features
        ('feature_selector', RFE(regressor, n_features_to_select=n_features))  # Feature selection
    ])
    return pipeline, regressor


def objective(trial, X_train, y_train, X_test, y_test, n_features):
    """
    Objective function for Optuna to optimize hyperparameters based on QWK.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)  # L2 regularization
    }

    # Create pipeline and regressor
    pipeline, regressor = create_pipeline(params, n_features)

    # Preprocess the data using the pipeline (scaling and feature selection)
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    # Train the regressor without early stopping
    regressor.fit(X_train_transformed, y_train)

    # Make predictions
    predictions = regressor.predict(X_test_transformed)
    predictions_rounded = np.round(predictions).astype(int)
    y_test_rounded = np.round(y_test).astype(int)
    
    # Calculate QWK
    qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic')
    
    # Return negative QWK to minimize
    return -qwk

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    return model.predict(X)


def evaluate_model(model, X, y):
    """
    Evaluate the model using RMSE, MAE, Pearson Correlation, QWK, Precision, and Recall.
    """
    predictions = make_predictions(model, X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    pearson_corr, _ = pearsonr(y, predictions)

    # Round predictions and true values to the nearest integer
    predictions_rounded = np.round(predictions).astype(int)
    y_test_rounded = np.round(y).astype(int)

    # Map rounded values to discrete labels
    unique_values = np.unique(np.concatenate([y_test_rounded, predictions_rounded]))
    value_to_label = {value: idx for idx, value in enumerate(unique_values)}
    y_test_labels = np.array([value_to_label[val] for val in y_test_rounded])
    predictions_labels = np.array([value_to_label[val] for val in predictions_rounded])

    # Calculate QWK
    qwk = cohen_kappa_score(y_test_labels, predictions_labels, weights='quadratic')

    # Calculate precision and recall based on "within 2 points" logic
    within_2 = (np.abs(predictions_labels - y_test_labels) <= 2).astype(int)  # 1 if within 2 points, 0 otherwise
    ground_truth = np.ones_like(within_2)  # All true values are "correct" (1)

    precision = precision_score(ground_truth, within_2, average='binary', zero_division=0)
    recall = recall_score(ground_truth, within_2, average='binary', zero_division=0)
    return rmse, mae, pearson_corr, qwk, precision, recall, predictions

def predict_grade(prompt, essay_file, best_params, n_features):
    # Load the essay text
    with open(essay_file, 'r') as f:
        essay_text = f.read()

    # Preprocess the essay text
    essay_features = preprocess_essay_text(essay_text)

    # Create a pipeline with the best parameters
    pipeline, regressor = create_pipeline(best_params, n_features)

    # Transform the essay features
    essay_features_transformed = pipeline.transform(essay_features)

    # Predict the grade
    predicted_grade = regressor.predict(essay_features_transformed)

    print(f'Predicted grade for prompt "{prompt}" and essay "{essay_file}": {predicted_grade:.2f}')
    
def preprocess_essay_text(text):
    """
    Preprocess the text by tokenizing, removing stop words, etc.
    """
    # Example preprocessing steps (customize as needed):
    # Tokenization, removing stop words, etc.
    return text  # Replace with actual preprocessing

def main(args=None):
    # Load extracted features from feature_extract.py
    features = preprocess_data(load_data(r'.\Extracted Features\extracted_features.csv'))

    # Scale the score column to be out of 100
    features['score'] = features['score'] * (100/9)
    
    # Feature engineering
    X = features.drop(columns=['score'])
    y = features['score']  # Replace 'score' with the actual target column name
    
    # Perform 5-fold cross-validation
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    pearson_scores = []
    qwk_scores = []
    precision_scores = []
    recall_scores = []

    n_features = 10  # Number of features to select
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Use Optuna for Bayesian Optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test, n_features), n_trials=35)

        # Get the best parameters and train the final model
        best_params = study.best_params
        pipeline, regressor = create_pipeline(best_params, n_features)

        # Preprocess the data
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)

        # Train the final model without early stopping
        regressor.fit(X_train_transformed, y_train)

        # Evaluate the model
        rmse, mae, pearson_corr, qwk, precision, recall, predictions = evaluate_model(regressor, X_test_transformed, y_test)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Calculate the average metrics across all folds
    average_rmse = np.mean(rmse_scores)
    average_mae = np.mean(mae_scores)
    average_pearson = np.mean(pearson_scores)
    average_qwk = np.mean(qwk_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)

    print(f'Average Root Mean Squared Error ({folds}-Fold CV): {average_rmse}')
    print(f'Average Mean Absolute Error ({folds}-Fold CV): {average_mae}')
    print(f'Average Pearson Correlation ({folds}-Fold CV): {average_pearson}')
    print(f'Average QWK ({folds}-Fold CV): {average_qwk}')
    print(f'Average Precision ({folds}-Fold CV): {average_precision}')
    print(f'Average Recall ({folds}-Fold CV): {average_recall}')
    

    # Define the metrics and their corresponding values
    metrics = [
        ("Average Root Mean Squared Error", average_rmse),
        ("Average Mean Absolute Error", average_mae),
        ("Average Pearson Correlation", average_pearson),
        ("Average QWK", average_qwk),
        ("Average Precision", average_precision),
        ("Average Recall", average_recall)
    ]

    # Write to CSV
    with open('xgboost_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(["Metric", "Value"])
        
        # Write metric data
        writer.writerows(metrics)

    print("CSV file saved successfully!")
    
    # argparse for command line arguments
    if args is not None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--prompt", help="The prompt for the essay")
        parser.add_argument("-e", "--essay_file", help="The essay text file")
        args = parser.parse_args()
    
    #return a grade if user prompts an essay and prompt
    if args.prompt and args.essay_file:
        predict_grade(args.prompt, args.essay_file, best_params, n_features)
    
if __name__ == "__main__":
    main()