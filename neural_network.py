import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr
import pandas as pd
import time
import optuna
import warnings
# Suppress warnings from PyTorch and Optuna
warnings.filterwarnings("ignore", category=UserWarning)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.2, 
                 hidden_dims=[128, 64], use_batch_norm=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Define fully connected layers with dynamic hidden dimensions
        fc_layers = []
        
        # Input to first hidden layer (from RNN output)
        rnn_output_size = hidden_size * 2  # Bidirectional => *2
        
        if hidden_dims:
            # Input layer
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(rnn_output_size))
            fc_layers.append(nn.Linear(rnn_output_size, hidden_dims[0]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                if use_batch_norm:
                    fc_layers.append(nn.BatchNorm1d(hidden_dims[i]))
                fc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(hidden_dims[-1]))
            fc_layers.append(nn.Linear(hidden_dims[-1], 1))
        else:
            # Direct connection to output if no hidden dims provided
            fc_layers.append(nn.Linear(rnn_output_size, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Reshape input for RNN if needed: [batch_size, seq_len, input_size]
        # For non-sequential data, we treat each feature as a time step
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # Pass through RNN
        rnn_out, _ = self.rnn(x)
        
        # Take the output of the last time step
        rnn_out = rnn_out[:, -1, :]
        
        # Pass through fully connected layers
        out = self.fc_layers(rnn_out)
        
        return out
    
class NeuralNetworkPipeline:
    def __init__(self, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=100,
                 batch_size=32, dropout_rate=0.2, hidden_dims=[128, 64, 32],
                 use_batch_norm=True, weight_decay=0.01, loss_fn=nn.MSELoss()):
        """
        Initialize the pipeline with hyperparameters for the neural network.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.scaler = StandardScaler()
        self.model = None
        self.opt = None
        self.best_model_state = None
        
    def fit(self, X_train, y_train, val_size=0.2, patience=10, verbose=True):
        """
        Train the neural network model with scaled data and validation.
        """
        # Split into training and validation sets
        if val_size > 0:
            X_train_set, X_val_set, y_train_set, y_val_set = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42
            )
        else:
            X_train_set, X_val_set = X_train, None
            y_train_set, y_val_set = y_train, None

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_set)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_set.values, dtype=torch.float32).reshape(-1, 1)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data if available
        if val_size > 0:
            X_val_scaled = self.scaler.transform(X_val_set)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val_set.values, dtype=torch.float32).reshape(-1, 1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        input_size = X_train_scaled.shape[1]
        self.model = RNN(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            hidden_dims=self.hidden_dims,
            use_batch_norm=self.use_batch_norm
        )
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        # Initialize early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for x, y in train_dataloader:
                self.opt.zero_grad()
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.opt.step()
                epoch_loss += loss.item() * x.size(0)
            
            avg_train_loss = epoch_loss / len(train_dataset)
            
            # Validation phase
            if val_size > 0:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for x, y in val_dataloader:
                        outputs = self.model(x)
                        loss = self.loss_fn(outputs, y)
                        val_loss += loss.item() * x.size(0)
                
                avg_val_loss = val_loss / len(val_dataset)
                
                # Learning rate scheduler step
                scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0
                    # Save the best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    early_stopping_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], '
                          f'Train Loss: {avg_train_loss:.4f}, '
                          f'Val Loss: {avg_val_loss:.4f}')
                
                if early_stopping_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}')
        
        # Load the best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        end_time = time.time()
        if verbose:
            print(f"Training finished in {end_time - start_time:.2f} seconds.")
            
    def predict(self, X_test):
        """
        Make predictions using the trained model.
        """
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            test_dataset = TensorDataset(X_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size * 2)
            
            for (batch_X,) in test_dataloader:
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
                
        predictions = np.concatenate(predictions, axis=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using multiple metrics.
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mae = mean_absolute_error(y_test, predictions)
        pearson_corr, p_val = pearsonr(y_test, predictions)
        
        # Calculate Huber loss
        y_test_tensor = torch.tensor(y_test.values).reshape(-1, 1)
        predictions_tensor = torch.tensor(predictions).reshape(-1, 1)
        huber_loss_fn = nn.HuberLoss(delta=1.0)
        
        with torch.no_grad():
            huber_loss = huber_loss_fn(predictions_tensor, y_test_tensor).cpu().item()
        
        # Calculate Quadratic Weighted Kappa (QWK)
        predictions_rounded = np.round(predictions).astype(int)
        y_test_rounded = np.round(y_test).astype(int)
        unique_labels = np.unique(np.concatenate([y_test_rounded, predictions_rounded]))
        qwk = cohen_kappa_score(y_test_rounded, predictions_rounded, weights='quadratic', labels=unique_labels)
        
        return rmse, mae, huber_loss, pearson_corr, qwk, predictions

def objective(trial, X, y, cv_folds=3):
    """
    Objective function for Optuna optimization.
    """
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    
    # Dynamic hidden dimensions
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)
    hidden_dims = []
    for i in range(n_hidden_layers):
        hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 16, 256))
    
    # Batch normalization
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Cross-validation setup
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline = NeuralNetworkPipeline(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=50,  # Reduced for optimization speed
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
            weight_decay=weight_decay
        )
        
        # Train with no validation split (already split via KFold)
        pipeline.fit(X_train_fold, y_train_fold, val_size=0, verbose=False)
        
        # Evaluate
        rmse, _, _, _, _, _ = pipeline.evaluate(X_val_fold, y_val_fold)
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)

def hyperparameter_tuning(X, y, n_trials=100, study_name='rnn_optimization'):
    """
    Perform hyperparameter tuning using Optuna.
    """
    study = optuna.create_study(direction='minimize', study_name=study_name)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

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
    # Load extracted features
    features = preprocess_data(load_data(r'.\Extracted Features\extracted_features.csv'))
    
    # Scale the score column to be out of 100
    features['score'] = features['score'] * (100/9)
    
    # Feature engineering
    X = features.drop('score', axis=1)
    y = features['score']
    
    # Perform hyperparameter tuning
    print("Starting hyperparameter tuning...")
    best_params = hyperparameter_tuning(X, y, n_trials=20)  # Reduced for demo
    
    # Extract the best hyperparameters
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    dropout_rate = best_params['dropout_rate']
    weight_decay = best_params['weight_decay']
    use_batch_norm = best_params['use_batch_norm']
    
    # Construct hidden_dims from the best parameters
    hidden_dims = []
    for i in range(best_params['n_hidden_layers']):
        hidden_dims.append(best_params[f'hidden_dim_{i}'])
    
    print(f"Best hidden dimensions: {hidden_dims}")
    
    # Perform 5-fold cross-validation with the best parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    mae_scores = []
    huber_scores = []
    pearson_scores = []
    qwk_scores = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold+1}/5...")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize and train the pipeline with best parameters
        nn_pipeline = NeuralNetworkPipeline(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            epochs=100,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
            weight_decay=weight_decay
        )
        
        nn_pipeline.fit(X_train, y_train)
        
        # Evaluate the pipeline
        rmse, mae, huber_loss, pearson_corr, qwk, _ = nn_pipeline.evaluate(X_test, y_test)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        huber_scores.append(huber_loss)
        pearson_scores.append(pearson_corr)
        qwk_scores.append(qwk)
        
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, QWK: {qwk:.4f}")
    
    # Calculate the average evaluation metrics across all folds
    average_rmse = np.mean(rmse_scores)
    average_mae = np.mean(mae_scores)
    average_huber = np.mean(huber_scores)
    average_pearson = np.mean(pearson_scores)
    average_qwk = np.mean(qwk_scores)
    
    print("\nFinal Cross-Validation Results:")
    print(f'Average Root Mean Squared Error (5-Fold CV): {average_rmse:.4f}')
    print(f'Average Mean Absolute Error (5-Fold CV): {average_mae:.4f}')
    print(f'Average Huber Loss (5-Fold CV): {average_huber:.4f}')
    print(f'Average Pearson Correlation (5-Fold CV): {average_pearson:.4f}')
    print(f'Average QWK (5-Fold CV): {average_qwk:.4f}')

if __name__ == "__main__":
    main()