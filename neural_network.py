import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import pandas as pd
import time

class NN(nn.Module): 
    def __init__(self, input_size, act = nn.ReLU(), dropout_rate = 0.2):
        super(NN, self).__init__()
        self.input_size = input_size
        self.act = act
        self.dropout_rate = dropout_rate

        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout(self.act(self.l1(x)))
        x = self.dropout(self.act(self.l2(x)))
        x = self.out(x)
        return x
    
class NeuralNetworkPipeline:
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=32, dropout_rate=0.2, act=nn.ReLU(), loss = nn.MSELoss()):
        """
        Initialize the pipeline with hyperparameters for the neural network.
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.opt = None
        self.loss = loss
        self.act = act
        self.dropout_rate = dropout_rate

    def fit(self, X_train, y_train):
        """
        Train the neural network model with scaled data.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = X_train_scaled.shape[1]
        self.model = NN(input_size=input_size, act=self.act, dropout_rate=self.dropout_rate)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

        # Add L2 regularization (weight decay)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # Initialize early stopping
        patience = 10
        best_loss = float('inf')
        early_stopping_counter = 0

        self.model.train()
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for x, y in dataloader:
                self.opt.zero_grad()
                outputs = self.model(x)
                err = self.loss(outputs, y)
                err.backward()
                self.opt.step()

                epoch_loss += err.item() * x.size(0)

            avg_epoch_loss = epoch_loss / len(dataset)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')

            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")



    def predict(self, X_test):
        """
        Make predictions using the trained model.
        """
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.model.eval() 
        predictions_np = []
        with torch.no_grad(): 
            test_dataset = TensorDataset(X_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size * 2) 
            for (batch_X,) in test_dataloader:
                 outputs = self.model(batch_X)
                 predictions_np.append(outputs.cpu().numpy()) 

        predictions_np = np.concatenate(predictions_np, axis=0)
        return predictions_np.flatten()

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Mean Squared Error (MSE).
        """
        predictions = self.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = mean_absolute_error(y_test, predictions)
        pearson_corr, p_val = pearsonr(y_test, predictions)

        #huber
        y_test_tensor = torch.tensor(y_test.values).reshape(-1, 1)
        predictions_tensor = torch.tensor(predictions).reshape(-1, 1)
        huber_loss_fn = nn.HuberLoss(delta=1.0) 
        with torch.no_grad():
            huber_loss = huber_loss_fn(predictions_tensor, y_test_tensor).cpu().item()

        #QWK
        predictions_rounded = np.round(predictions).astype(int)
        y_test_rounded = np.round(y_test).astype(int)
        unique_labels = np.unique(np.concatenate([y_test_rounded, predictions_rounded]))
        qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic', labels=unique_labels)

        return rmse, mae, huber_loss, pearson_corr, qwk, predictions
    
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
    #load extracted features from feature_extract.py
    features = preprocess_data(load_data(r'.\Extracted Features\extracted_features.csv'))
    # Scale the score column to be out of 100# Scale the score column to be out of 100
    features['score'] = features['score'] * (100/9)
    # Feature engineering
    X = features.drop('score', axis=1)  # Replace 'target_column' with the actual target column name
    y = features['score']  # Replace 'target_column' with the actual target column name

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    huber_scores = []
    pearson_scores = []
    qwk_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize and train the pipeline
        nn_pipeline = NeuralNetworkPipeline(learning_rate=0.001, epochs=100, batch_size=32, act=nn.ReLU(), loss = nn.MSELoss())
        nn_pipeline.fit(X_train, y_train)

        # Evaluate the pipeline
        rmse, mae, huber_loss, pearson_corr, qwk, predictions = nn_pipeline.evaluate(X_test, y_test)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        huber_scores.append(huber_loss)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)

    # Calculate the average evaluation metrics across all folds
    average_rmse = np.mean(rmse_scores)
    average_mae = np.mean(mae_scores)
    average_huber = np.mean(huber_scores)
    average_pearson = np.mean(pearson_scores)
    average_qwk = np.mean(qwk_scores)

    print(f'Average Root Mean Squared Error (5-Fold CV): {average_rmse}')
    print(f'Average Mean Absolute Error (5-Fold CV): {average_mae}')
    print(f'Average Huber Loss (5-Fold CV): {average_huber}')
    print(f'Average Pearson Correlation (5-Fold CV): {average_pearson}')
    print(f'Average QWK (5-Fold CV): {average_qwk}')

if __name__ == "__main__":
    main()