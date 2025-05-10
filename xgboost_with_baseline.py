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
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump, load
import os
from sklearn.model_selection import train_test_split
import warnings

def save_results(model, pipeline, best_params, metrics, feature_names, filepath="model_results.pkl"):
    results = {
        'model': model,
        'pipeline': pipeline,
        'best_params': best_params,
        'metrics': metrics,
        'feature_names': feature_names,
    }
    directory = os.path.dirname(filepath)
    if directory: 
        os.makedirs(directory, exist_ok=True)
    dump(results, filepath)
    print(f"Results saved to {filepath}")

def load_results(filepath="model_results.pkl"):
    if os.path.exists(filepath):
        results = load(filepath)
        print(f"Results loaded from {filepath}")
        return results
    else:
        print(f"No saved results found at {filepath}")
        return None


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def calculate_qwk(y_true, y_pred, min_val=1, max_val=9):
    y_true_original = y_true * (max_val - min_val) / 100 + min_val
    y_pred_original = y_pred * (max_val - min_val) / 100 + min_val
    
    y_pred_rounded = np.round(y_pred_original * 2) / 2
    
    y_pred_clipped = np.clip(y_pred_rounded, min_val, max_val)
    y_true_clipped = np.clip(y_true_original, min_val, max_val)
    labels_true = (y_true_clipped * 2).astype(int)
    labels_pred = (y_pred_clipped * 2).astype(int)
    
    qwk = cohen_kappa_score(labels_true, labels_pred, weights='quadratic')

    return qwk

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
    if n_features is not None:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selector', RFE(regressor, n_features_to_select=n_features))
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler())
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
    qwk = calculate_qwk(y_test, predictions)
    
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
    qwk = calculate_qwk(y, predictions)

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


def feature_ablation_study(features, y, best_params):
    # Define different feature sets
    feature_sets = {
    "All Features": features.iloc[:, :-1],  
    "Only Linguistic Features": features[['noun_count', 'verb_count', 'adj_count', 'adv_count', 
                                        'pronoun_count', 'modal_count', 'complexity_verb_ratio', 
                                        'adj_adv_ratio', 'num_grammatical_errors']],
    "Only Readability Features": features[['avg_sentence_length', 'avg_syllables_per_word', 
                                         'flesch_reading_ease', 'unique_word_ratio']],
    "Only Semantic Features": features[['tfidf_cosine_similarity', 'sbert_prompt_adherence_similarity',
                                      'mean_tfidf_score', 'std_tfidf_score']],
    "Only Repetition Features": features[['window_repeat_count']],
    "Only Error Features": features[['num_grammatical_errors']]
}
    results = {}
    for name, X in feature_sets.items():
        print(f"\nEvaluating with {name}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if X.shape[1] == 1:
            scaler = StandardScaler()
            X_train_transformed = scaler.fit_transform(X_train.values.reshape(-1, 1))
            X_test_transformed = scaler.transform(X_test.values.reshape(-1, 1))
            
            regressor = XGBRegressor(objective='reg:squarederror',random_state=42,**best_params)
        else:
            pipeline, regressor = create_pipeline(best_params, min(10, X.shape[1]))
            
            X_train_transformed = pipeline.fit_transform(X_train, y_train)
            X_test_transformed = pipeline.transform(X_test)
        
        regressor.fit(X_train_transformed, y_train)
        predictions = regressor.predict(X_test_transformed)
        qwk = calculate_qwk(y_test, predictions)
        
        
        
        results[name] = qwk
        print(f"{name} - QWK: {results[name]:.3f}")
    
    return results


def model_ablation_study(X, y, best_params):
    models = {
        "XGBoost (Current)": XGBRegressor(**best_params),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10),
        "Linear Regression": Ridge(alpha=1.0),
        "Neural Network": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True))
    }
    
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        qwk = calculate_qwk(y_test, predictions)   
        results[name] = qwk           
        print(f"{name} - QWK: {qwk:.3f}")
    return results

def plot_ablation_bar(results, title="Ablation Study Results", color_palette="viridis"):
    plt.figure(figsize=(10, 6)) 
    ax = sns.barplot(x=list(results.values()), y=list(results.keys()), palette="viridis",edgecolor="black")
    for i, v in enumerate(results.values()):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    plt.title(title, fontweight='bold')
    plt.xlabel('QWK Score')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show() 

def plot_learning_curves(models, X_train, y_train, X_test, y_test, best_params, n_points=20):
    plt.figure(figsize=(12, 8), facecolor='#f5f5f5')
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    models = {
        'XGBoost': XGBRegressor(**best_params),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Neural Network': make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=0.001, 
                max_iter=1000,
                early_stopping=True,
                random_state=42
            )
        ),
        'Linear Regression': Ridge(alpha=1.0)
    }
    colors = {
        'XGBoost': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'Neural Network': '#2ca02c',
        'Linear Regression': '#d62728'
    }
    for name, model in models.items():
        train_scores = []
        val_scores = []
        train_sizes = []
        
        for size in np.logspace(0, 1, num=n_points, base=10):
            size = int(size/10 * len(X_train))
            size = max(10, min(size, len(X_train)))  
            
            X_subset = X_train[:size]
            y_subset = y_train[:size]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_subset, y_subset)
            
            # Calculate QWK instead of RMSE
            train_pred = model.predict(X_subset)
            val_pred = model.predict(X_test)
            
            train_scores.append(calculate_qwk(y_subset, train_pred))
            val_scores.append(calculate_qwk(y_test, val_pred))
            train_sizes.append(size)
        
        plt.plot(train_sizes, val_scores, '-',
                color=colors[name], 
                linewidth=3, label=f'{name} (Val)')
        
        final_qwk = val_scores[-1]
        plt.annotate(f'{name}: {final_qwk:.3f}',
                   xy=(0.98, 0.85 - 0.05*list(models.keys()).index(name)),
                   xycoords='axes fraction',
                   color=colors[name],
                   fontsize=11,
                   ha='right')

    plt.xscale('log')
    plt.xlabel('Training Examples (log scale)', fontsize=12)
    plt.ylabel('Quadratic Weighted Kappa (QWK)', fontsize=12)
    plt.title('Model Learning Curves (QWK Metric)', fontsize=14, pad=20)
    plt.legend(fontsize=10, framealpha=0.9)
    
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('qwk_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_quality(y_true, y_pred, max_val=9):
    y_true_orig = y_true * max_val / 100
    y_pred_orig = y_pred * max_val / 100
    
    plt.figure(figsize=(10, 8), facecolor='#f5f5f5')
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    scatter = plt.scatter(y_true_orig, y_pred_orig, alpha=0.6, 
                         c=np.abs(y_true_orig - y_pred_orig), 
                         cmap='viridis', s=80)
    
    min_val, max_val = min(min(y_true_orig), min(y_pred_orig)), max(max(y_true_orig), max(y_pred_orig))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
    
    plt.fill_between([min_val, max_val], [min_val-0.5, max_val-0.5], [min_val+0.5, max_val+0.5], 
                    color='#2ecc71', alpha=0.15, label='±0.5 points')
    plt.fill_between([min_val, max_val], [min_val-1, max_val-1], [min_val+1, max_val+1], 
                    color='#3498db', alpha=0.1, label='±1.0 points')
    
    within_half = sum(abs(y_true_orig - y_pred_orig) <= 0.5) / len(y_true)
    within_one = sum(abs(y_true_orig - y_pred_orig) <= 1.0) / len(y_true)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    
    plt.title(f'Prediction Accuracy\n{within_half:.1%} within ±0.5, {within_one:.1%} within ±1.0', 
              fontsize=14, pad=20)
    plt.xlabel('True Score', fontsize=12)
    plt.ylabel('Predicted Score', fontsize=12)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_quality.png', dpi=300, bbox_inches='tight')
    plt.close()

    return within_half, within_one

def plot_feature_importance(model, feature_names):
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    top_n = len(importances)
    
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(12, 8), facecolor='#f5f5f5')
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    bars = plt.barh(range(top_n), importances[indices], color=colors, alpha=0.8)
    
    for i, (v, bar) in enumerate(zip(importances[indices], bars)):
        plt.text(bar.get_width() + 0.005, i, f"{v:.3f}", 
                color='black', va='center', fontsize=10)
    
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=11)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, pad=20)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

class TransformerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name  
        self.tokenizer = None
        self.model = None
        
    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        return self
    
    def transform(self, texts):
        if not isinstance(texts, pd.Series):
            texts = pd.Series(texts)
            
        inputs = self.tokenizer(texts.tolist(), return_tensors='pt', 
                              padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:,0,:].numpy()
    
    def get_params(self, deep=True):
        return {"model_name": self.model_name}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def transformer_baseline(df):
    X = df['clean_essay']
    y = df['Overall'] * (100/9)
    
    pipeline = make_pipeline(
        TransformerFeatures(),
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    
    preds = cross_val_predict(pipeline, X, y, cv=5, n_jobs=-1)
    qwk = calculate_qwk(y, preds)
    
    print(f"Transformer Baseline - QWK: {qwk:.3f}")
    return qwk

def main(args=None):
    features = preprocess_data(load_data(r'.\Extracted Features\extracted_features.csv'))
    original_data = pd.read_csv("./data/ielts_data.csv")
    original_data = original_data.dropna()
    
    features['score'] = features['score'] * (100/9)
    
    X = features.iloc[:, :-1] 
    y = features['score']
    scaler = StandardScaler()
    scaler.fit_transform(X)
    
    saved_results = load_results()

    # Perform 80-10-10 train-test-validation split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    if saved_results:
        regressor = saved_results['model']
        pipeline = saved_results['pipeline']
        best_params = saved_results['best_params']
        metrics = saved_results['metrics']
        feature_names = saved_results['feature_names']
        
        print("Using saved model. Metrics:")
        for name, value in metrics:
            print(f"{name}: {value:.4f}")

    else:
        rmse_scores = []
        mae_scores = []
        pearson_scores = []
        qwk_scores = []
        precision_scores = []
        recall_scores = []

        n_features = 10  
        best_params = None 

        # Use Optuna for Bayesian Optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, n_features), n_trials=35)

        best_params = study.best_params
        pipeline, regressor = create_pipeline(best_params, n_features)

        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)

        regressor.fit(X_train_transformed, y_train)

        rmse, mae, pearson_corr, qwk, precision, recall, predictions = evaluate_model(regressor, X_test_transformed, y_test)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)
        precision_scores.append(precision)
        recall_scores.append(recall)

        average_rmse = np.mean(rmse_scores)
        average_mae = np.mean(mae_scores)
        average_pearson = np.mean(pearson_scores)
        average_qwk = np.mean(qwk_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)

        print(f'Average Root Mean Squared Error: {average_rmse}')
        print(f'Average Mean Absolute Error: {average_mae}')
        print(f'Average Pearson Correlation: {average_pearson}')
        print(f'Average QWK: {average_qwk}')
        print(f'Average Precision: {average_precision}')
        print(f'Average Recall: {average_recall}')

        metrics = [
            ("Average Root Mean Squared Error", average_rmse),
            ("Average Mean Absolute Error", average_mae),
            ("Average Pearson Correlation", average_pearson),
            ("Average QWK", average_qwk),
            ("Average Precision", average_precision),
            ("Average Recall", average_recall)
        ]
        save_results(regressor, pipeline, best_params, metrics, X.columns.tolist())
        
    print("\nRunning feature ablation study...")
    feature_results = feature_ablation_study(features, y, best_params)

    print("\nRunning model ablation study...")
    model_results = model_ablation_study(X, y, best_params)
    
    try:
        X_test_transformed = pipeline.transform(X_test)
    except ValueError as e:
        print(f"Error transforming data: {e}")
        print("Trying to refit the feature selector...")
        pipeline.fit(X_train, y_train)  
        X_test_transformed = pipeline.transform(X_test)
    models = {
        'XGBoost': XGBRegressor(**best_params),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10),
        'Neural Network': make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)
        ),
        'Linear Regression': Ridge(alpha=1.0)
    }
    plot_learning_curves(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        best_params = best_params,
        n_points=15  
    )
    y_pred = regressor.predict(X_test_transformed)
    plot_prediction_quality(y_test, y_pred)
    plot_feature_importance(regressor, feature_names)

    plot_ablation_bar(feature_results, title="Feature Ablation Study (QWK)")
    plot_ablation_bar(model_results, title="Model Comparison (QWK)")
        
if __name__ == "__main__":
    main()