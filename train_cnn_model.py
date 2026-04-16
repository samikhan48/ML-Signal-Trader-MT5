import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1) # [Batch, Features, Seq_Len]
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TemporalCNN(nn.Module):
    def __init__(self, num_features, seq_len):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc = nn.Linear(64 * (seq_len // 4), 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_sequences(data, targets, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = targets[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_cnn(symbol_csv, symbol_name):
    logging.info(f"--- Training CNN for {symbol_name} ---")
    
    if not os.path.exists(symbol_csv):
        logging.error(f"File {symbol_csv} not found!")
        return

    df = pd.read_csv(symbol_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Feature Engineering (Simplified)
    df['Returns'] = df['Close'].pct_change()
    df['Range'] = (df['High'] - df['Low']) / df['Close']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Add some indicators
    df['RSI'] = 100 - (100 / (1 + df['Returns'].rolling(14).mean().abs() / df['Returns'].rolling(14).std()))
    df['SMA_Ratio'] = df['Close'].rolling(10).mean() / df['Close'].rolling(30).mean()
    
    df = df.dropna()
    
    # Features
    feature_cols = ['Returns', 'Range', 'RSI', 'SMA_Ratio']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[feature_cols])
    targets = df['Target'].values
    
    SEQ_LEN = 24
    X, y = create_sequences(data_scaled, targets, SEQ_LEN)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    train_dataset = TradingDataset(X_train, y_train)
    test_dataset = TradingDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = TemporalCNN(len(feature_cols), SEQ_LEN).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_test, all_preds)
    logging.info(f"Final Test Accuracy: {acc:.2%}")
    
    # Save Model & Scaler
    model_path = f"cnn_model_{symbol_name}.pth"
    torch.save(model.state_dict(), model_path)
    
    scaler_path = f"cnn_scaler_{symbol_name}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols,
            'seq_len': SEQ_LEN
        }, f)
        
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    # train_cnn("mt5_data_EURUSD_H1.csv", "EURUSD")
    # train_cnn("mt5_data_GBPUSD_H1.csv", "GBPUSD")
    train_cnn("mt5_data_USDJPY_H1.csv", "USDJPY")
    train_cnn("mt5_data_AUDUSD_H1.csv", "AUDUSD")
