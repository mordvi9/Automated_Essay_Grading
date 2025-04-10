import pandas as pd
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import numpy as np

class SupportVectorRegressionModel:
    def __init__(self):
        #TODO include the NLP features inside the pipeline
        self.model = make_pipeline(StandardScaler(), SVR())
    
    def train(self, X_train, y_train):
        #TODO explore expanded hyperparameter space
        param_grid = {
            'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 0.5]
        }
        grid_search = RandomizedSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10, random_state=42)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        huber = tf.keras.losses.Huber(delta=1.0)
        huber_loss = huber(y_test, predictions).numpy().mean()
        pearson_corr, p_val = pearsonr(y_test, predictions)
        predictions_rounded = np.round(predictions).astype(int)
        y_test_rounded = np.round(y_test).astype(int)
        qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic', labels=list(range(10)))

        return mse, mae, huber_loss, pearson_corr, qwk, predictions

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
    #TODO add more data and preprocessing steps
    data = load_data('./data/ielts_data.csv')
    data = preprocess_data(data)
    
    #TODO Feature engineering
    X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name
    y = data['target_column']  # Replace 'target_column' with the actual target column name

    # Initialize the model
    svr_model = SupportVectorRegressionModel()

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
        
        # Train the model
        svr_model.train(X_train, y_train)
        
        # Evaluate the model
        mse, mae, huber_loss, pearson_corr, qwk, predictions = svr_model.evaluate(X_test, y_test)
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

if __name__ == "__main__":
    main()