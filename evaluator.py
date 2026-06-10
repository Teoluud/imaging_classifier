import torch
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader

from logger import logger


class Evaluator:
    """ Handles the evaluation of a trained model on a test dataset.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            accuracy_fn: MulticlassAccuracy,
            device: torch.device
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn.to(device)
        self.device = device

    def evaluate(self, data_loader: DataLoader, split_name: str = "Test") -> dict[str, float]:
        """ Runs a full forward pass over the data_loader and computes evaluation metrics.
        """
        logger.debug(f"Starting evaluation on {split_name} set...")

        eval_loss, eval_acc = 0, 0

        self.model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                # Put data to target device
                X, y = X.to(self.device), y.to(self.device)

                predictions = self.model(X)

                eval_loss += self.loss_fn(predictions, y).item()
                eval_acc += self.accuracy_fn(predictions, y).item()

        eval_loss /= len(data_loader)
        eval_acc /= len(data_loader)

        logger.info(f"[{split_name} Results] Loss: {eval_loss:.5f} | Accuracy: {eval_acc:.2%}")

        return {
            "loss": eval_loss,
            "accuracy": eval_acc
        }