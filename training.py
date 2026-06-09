import torch
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

from logger import logger


class TrainingLoop:
    """ Class that handles the training loop.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 accuracy_fn: MulticlassAccuracy,
                 device: torch.device) -> None:
        """ Constructor.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn.to(device)
        self.device = device

    def train_step(self, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """ Performs a training step with model trying to learn on data_loader.
        """
        train_loss, train_acc = 0, 0
        self.model.train()
        for X, y in data_loader:
            # Put data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_pred = self.model(X)

            # Calculate loss and accuracy (per batch)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss
            train_acc = self.accuracy_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average training loss and accuracy
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        tqdm.write(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")

        return train_loss
    
    def validation_step(self, data_loader: torch.utils.data.DataLoader) -> float:
        """ Performs a validation loop step on model going over data_loader.
        """
        val_loss, val_acc = 0, 0
        
        self.model.eval()
        with torch.inference_mode():
            for X_val, y_val in data_loader:
                # Send data to target device
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                val_pred = self.model(X_val)

                # Calculate loss and accuracy (per batch)
                val_loss += self.loss_fn(val_pred, y_val)
                val_acc += self.accuracy_fn(val_pred, y_val)

            # Adjust metrics and print out
            val_loss /= len(data_loader)
            val_acc /= len(data_loader)
            tqdm.write(f"Validation loss: {val_loss:.5f} | Validation accuracy: {val_acc:.2f}%")
        
            return val_loss