import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryROC
from mlxtend.plotting import plot_confusion_matrix


def plot_training_results(
        epochs: int,
        train_losses: list[float],
        val_losses: list[float],
        learning_rates: list[float],
        save_path: str = "loss_curves.png"
) -> None:
    """ Generates and saves optimization loss metrics.
    """
    epoch_x = np.arange(0, epochs, 1)

    plt.figure(figsize=(12, 6))

    # Loss curves subplot
    plt.subplot(1, 2, 1)
    plt.title("Train and Validation Loss")
    plt.plot(epoch_x, np.array(train_losses), label="Train Loss")
    plt.plot(epoch_x, np.array(val_losses), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Learning rate subplot
    plt.subplot(1, 2, 2)
    plt.title("Learning Rate")
    plt.plot(epoch_x, np.array(learning_rates))
    plt.xlabel("Epoch")
    plt.ylabel("LR")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_conf_matrix(
        preds: torch.Tensor,
        truths: torch.Tensor,
        class_names: tuple[str, ...],
        save_path: str = "confusion_matrix.png"
) -> None:
    """ Generates and saves a multiclass confusion matrix.
    """
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
    confmat_tensor = confmat(preds=preds, target=truths)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7)
    )

    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
        probs: torch.Tensor,
        truths: torch.Tensor,
        save_path: str = "roc_curve.png"
) -> None:
    """ Generates and saves a Binary ROC Curve, evaluating the positive class.
    """
    roc = BinaryROC()
    # probs[:, 1] extracts the probabilities for the positive class (Electron)
    roc.update(preds=probs[:, 1], target=truths)

    fig, ax = roc.plot(score=True)
    fig.savefig(save_path)
    plt.close()
