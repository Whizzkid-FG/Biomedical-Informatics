import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Tuple, List, Dict

from uvicorn import Config
from utils.logger import setup_logger

logger = setup_logger('disease_predictor')

class PatientDataset(Dataset):
    """Custom Dataset for patient data."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class DiseasePredictor(nn.Module):
    """
    Enhanced neural network model for disease prediction with additional features.
    
    Features:
    - Multiple disease prediction
    - Attention mechanism for feature importance
    - Batch normalization for better training
    - Residual connections
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], num_diseases: int):
        super(DiseasePredictor, self).__init__()
        
        self.input_size = input_size
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(Config.DISEASE_MODEL_PARAMS['dropout_rate'])
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_size, num_diseases)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Forward through hidden layers with residual connections
        residual = x
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i % 4 == 0 and x.shape == residual.shape:  # Add residual every 4 layers if shapes match
                x = x + residual
                residual = x
        
        # Output probabilities
        output = torch.sigmoid(self.output_layer(x))
        return output, attention_weights

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = nn.BCELoss(),
        learning_rate: float = Config.DISEASE_MODEL_PARAMS['learning_rate']
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            learning_rate: Initial learning rate
            
        Returns:
            Dict containing training history
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(Config.DISEASE_MODEL_PARAMS['num_epochs']):
            # Training phase
            self.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                predictions, _ = self(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    predictions, _ = self(batch_features)
                    val_loss += criterion(predictions, batch_labels).item()
                    val_preds.extend(predictions.numpy())
                    val_labels.extend(batch_labels.numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.state_dict(), 'models/best_disease_predictor.pt')
            else:
                patience_counter += 1
                if patience_counter >= Config.DISEASE_MODEL_PARAMS['early_stopping_patience']:
                    logger.info(f'Early stopping triggered at epoch {epoch}')
                    break
            
            logger.info(
                f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
                f'Val Loss = {val_loss:.4f}, Val AUC = {val_auc:.4f}'
            )
        
        return history

    def predict_with_confidence(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores and feature importance.
        
        Args:
            features: Input features tensor
            threshold: Probability threshold for positive prediction
            
        Returns:
            Tuple containing predictions, probabilities, and feature importance
        """
        self.eval()
        with torch.no_grad():
            probabilities, attention_weights = self(features)
            predictions = (probabilities > threshold).float()
            
            return (
                predictions.numpy(),
                probabilities.numpy(),
                attention_weights.numpy()
            )