from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryROC
from mlxtend.plotting import plot_confusion_matrix


def build_file_list(parent_path: Path | str, prefix: str) -> list[Path]:
    """ (NOT IN USE) Generates the data file list.
    """
    parent_dir = Path(parent_path)
    file_names = []
    idx = 0
    while True:
        path = parent_dir / (prefix + str(idx) + ".hdf5")
        if not path.exists():
            break
        file_names.append(path)
        idx += 1
        
    return file_names


def plot_training_results(
        epochs: int,
        train_losses: list[float],
        val_losses: list[float],
        learning_rates: list[float],
        save_path: str | Path = "loss_curves.png",
        title: str = "Training Results"
) -> None:
    """ Generates and saves optimization loss metrics.
    """
    epoch_x = np.arange(0, epochs, 1) + 1

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)
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
        save_path: str | Path = "confusion_matrix.png",
        title: str = "Confusion Matrix"
) -> None:
    """ Generates and saves a multiclass confusion matrix.
    """
    confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
    confmat_tensor = confmat(preds=preds, target=truths)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7),
        show_normed=True
    )

    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
        probs: torch.Tensor,
        truths: torch.Tensor,
        save_path: str | Path = "roc_curve.png",
        title: str = "ROC Curve"
) -> None:
    """ Generates and saves a Binary ROC Curve, evaluating the positive class.
    """
    roc = BinaryROC()
    # probs[:, 1] extracts the probabilities for the positive class (Electron)
    roc.update(preds=probs[:, 1], target=truths)

    fig, ax = roc.plot(score=True)
    plt.title(title)
    fig.savefig(save_path)
    plt.close()


def normalize_image(tensor_data: torch.Tensor, event_energy_mev: float) -> torch.Tensor:
    """ Normalize the log of the energy of the event display image w.r.t. the log of the event reconstructed energy.
    """
    norm_tensor = torch.zeros_like(tensor_data)

    # Check for actual active pixels in tensor_data
    active_pixels = tensor_data > 0     # this is a mask, True when the condition is met, False when it's not.
    if active_pixels.any():
        active_kev = tensor_data[active_pixels] * 1000.0
        event_energy_kev = event_energy_mev * 1000.0
        log_norm_factor = np.log10(max(event_energy_kev, 1.0))  # to ensure positive normalization
        norm_tensor[active_pixels] = torch.log10(active_kev) / log_norm_factor

    return norm_tensor


def normalize_merit(merit_vars: torch.Tensor) -> torch.Tensor:
    """ Normalize each merit variable to the maximum value in the dataset.
    """
    # Find the max values for each variable
    max_values = torch.amax(merit_vars, dim=0)
    min_values = torch.amin(merit_vars, dim=0)

    # Normalize
    norm_vars = (merit_vars - min_values) / (max_values - min_values)

    return norm_vars
