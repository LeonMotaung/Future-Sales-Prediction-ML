import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import TensorDataset, DataLoader

class TCGLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class TCGSalesPredictor:
    def __init__(self, sequence_length=30, prediction_horizon=7):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        
        # TCG constants adapted for sales patterns
        self.tau = 0.561  # For trend smoothing
        self.iota = complex(-0.5, 0.866)  # For seasonal patterns
        self.kappa = 0.8  # For promotional impact
        
    def create_tcg_features(self, sales_data, features_df=None):
        """
        Create TCG-enhanced features for sales prediction
        """
        features = []
        
        # Basic sales features
        features.append(sales_data)  # Original sales
        
        # Tau features - trend and smooth perturbations
        rolling_mean = sales_data.rolling(window=7).mean()
        tau_trend = sales_data * self.tau + (1-self.tau) * rolling_mean
        features.append(tau_trend.fillna(method='bfill'))
        
        # Iota features - seasonal and cyclical patterns
        # Real part for weekly seasonality
        weekly_seasonal = sales_data * np.real(self.iota) 
        # Imaginary part for monthly patterns
        monthly_pattern = sales_data.rolling(window=30).mean() * np.imag(self.iota)
        features.extend([weekly_seasonal.fillna(0), monthly_pattern.fillna(0)])
        
        # Kappa features - promotional impact and volatility
        sales_volatility = sales_data.rolling(window=14).std()
        kappa_impact = np.power(np.abs(sales_data), 0.25) * self.kappa * (1 + sales_volatility)
        features.append(kappa_impact.fillna(0))
        
        # Lag features with TCG transformation
        for lag in [1, 7, 14, 30]:
            lag_feature = sales_data.shift(lag)
            tcg_lag = lag_feature * self.tau + np.real(self.iota) * lag_feature.rolling(7).mean()
            features.append(tcg_lag.fillna(0))
        
        return np.column_stack(features)

class TCGSalesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.tau = nn.Parameter(torch.tensor(0.561))
        self.iota_real = nn.Parameter(torch.tensor(0.5))
        self.iota_imag = nn.Parameter(torch.tensor(0.866))
        
        # TCG-enhanced LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        
        # TCG attention mechanism
        self.attention = TCGAttention(hidden_dim)
        
        # Output layers with TCG stabilization
        self.fc1 = TCGLinear(hidden_dim, hidden_dim // 2)
        self.fc_out = TCGLinear(hidden_dim // 2, 3)  # For classification
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # TCG attention for important time steps
        attended_out, attention_weights = self.attention(lstm_out)
        
        # Use the output of the last time step for classification
        last_time_step_out = attended_out[:, -1, :]

        # TCG-enhanced prediction
        features = torch.relu(self.fc1(last_time_step_out))
        features = self.dropout(features)
        
        # Classification output
        output = self.fc_out(features)
        
        return output, attention_weights

class TCGAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = TCGLinear(hidden_dim, hidden_dim)
        self.key = TCGLinear(hidden_dim, hidden_dim)
        self.value = TCGLinear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention with TCG stabilization
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_output = torch.matmul(attention_weights, V)
        
        return attended_output, attention_weights

class TCGClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2, epochs=10, batch_size=32, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = TCGSalesModel(input_dim, hidden_dim, num_layers, dropout)
        self.classes_ = np.array([0, 1, 2]) # Corresponds to Low, Medium, High

    def fit(self, X, y):
        # Ensure y is a numpy array
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()
