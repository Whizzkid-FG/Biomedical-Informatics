import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from sklearn.metrics import roc_auc_score
from config.config import Config

class DiseasePredictor(nn.Module):
    """Neural network model for disease prediction."""

    def __init__(self, input_size: int, hidden_layers: List[int], num_classes: int):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(Config.DISEASE_MODEL_PARAMS['dropout_rate'])
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_model(self, train_loader, val_loader, num_epochs: int = 100) -> Dict[str, List[float]]:
        optimizer = optim.Adam(self.parameters(), lr=Config.DISEASE_MODEL_PARAMS['learning_rate'])
        criterion = nn.BCELoss()
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                predictions = self(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss, val_preds, val_labels = 0.0, [], []
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    predictions = self(batch_features)
                    val_loss += criterion(predictions, batch_labels).item()
                    val_preds.extend(predictions.numpy())
                    val_labels.extend(batch_labels.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_auc'].append(val_auc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return history
