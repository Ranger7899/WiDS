import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --- Data Preprocessing ---
def preprocess_data(df):
    # Identify numerical and categorical columns (excluding targets)
    # Assuming target columns are 'Sex' and 'ADHD'
    target_cols = ['Sex', 'ADHD']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing values: median for numerical features, mode for categorical features
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        # Convert categorical columns to numeric codes
        df[col] = df[col].astype('category').cat.codes
    return df

# --- Custom Dataset for Multi-Task Learning ---
class MultiTaskDataset(Dataset):
    def __init__(self, df, target_cols):
        """
        df: pandas DataFrame containing the data
        target_cols: list of column names for the targets (e.g., ['Sex', 'ADHD'])
        """
        self.y = df[target_cols].values.astype(np.float32)
        self.X = df.drop(columns=target_cols).values.astype(np.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # y is a vector of targets

# --- Multi-Task Neural Network ---
class MultiTaskNN(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskNN, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        # Separate heads for each task
        self.sex_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability for sex classification
        )
        self.adhd_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability for ADHD diagnosis
        )
    
    def forward(self, x):
        shared_rep = self.shared(x)
        sex_pred = self.sex_head(shared_rep)
        adhd_pred = self.adhd_head(shared_rep)
        # Concatenate the two outputs along the last dimension
        return torch.cat([sex_pred, adhd_pred], dim=1)

# --- Training Loop ---
def main():
    # Load data (update the path as needed)
    data = pd.read_csv('../input/train.csv')
    data = preprocess_data(data)
    
    # Define target columns (update as necessary)
    target_cols = ['Sex', 'ADHD']  # adjust to match your dataset
    
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = MultiTaskDataset(train_df, target_cols)
    val_dataset = MultiTaskDataset(val_df, target_cols)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_df.drop(columns=target_cols).shape[1]
    model = MultiTaskNN(input_dim).to(device)
    
    # Define loss functions (one for each task) and optimizer
    criterion = nn.BCELoss()  # Binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            # Calculate loss for both tasks and average them
            loss_sex = criterion(preds[:, 0], y_batch[:, 0].unsqueeze(1))
            loss_adhd = criterion(preds[:, 1], y_batch[:, 1].unsqueeze(1))
            loss = (loss_sex + loss_adhd) / 2.0
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # --- Validation ---
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute AUC for each task
        auc_sex = roc_auc_score(all_targets[:, 0], all_preds[:, 0])
        auc_adhd = roc_auc_score(all_targets[:, 1], all_preds[:, 1])
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val AUC Sex: {auc_sex:.4f} - Val AUC ADHD: {auc_adhd:.4f}")

if __name__ == '__main__':
    main()
