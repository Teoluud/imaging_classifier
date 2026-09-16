from pathlib import Path
from typing import Any, cast

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray
import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryROC
from mlxtend.plotting import plot_confusion_matrix


def build_file_list(parent_path: Path | str, prefix: str) -> list[Path]:
    """ Generates the data file list. """
    parent_dir = Path(parent_path)
    file_names: list[Path] = []
    idx = 0
    while True:
        path = parent_dir / f"{prefix}{idx}.hdf5"
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
    """ Generates and saves optimization loss metrics. """
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

    _, _ = cast(
        tuple[Any, Any],
        plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(),
            class_names=class_names,
            figsize=(10, 7),
            show_normed=True,
        ),
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
    """ Generates and saves a Binary ROC Curve, evaluating the positive class. """

    roc = BinaryROC()
    # probs[:, 1] extracts the probabilities for the positive class (Electron)
    roc.update(preds=probs[:, 1], target=truths)

    fig, _ = cast(tuple[Any, Any], roc.plot(score=True))
    plt.title(title)
    fig.savefig(save_path)
    plt.close()


def plot_probs_distribution(
        probs: torch.Tensor,
        truths: torch.Tensor,
        save_path: str | Path = "probs_distribution.png",
        title: str = "Output probs distribution"
) -> None:
    """ Plots the distributions of the log(1 - probability), with the probability normalized between -1 (Proton) and 1 (Electron). """

    norm_probs = torch.log(1-(probs[:, 1] - probs[:, 0]))

    protons: list[float] = []
    electrons: list[float] = []
    for i, prob in enumerate(norm_probs):
        if truths[i] == 0:
            protons.append(prob.item())
        else:
            electrons.append(prob.item())

    plt.hist(protons, bins='auto', histtype='step', label='Protons', alpha=0.7)
    plt.hist(electrons, bins='auto', histtype='step', label='Electrons', alpha=0.7)
    plt.xlabel('log(1-p)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_energy_distribution(
        energies: NDArray[np.float32],
        truths: torch.Tensor,
        save_path: str | Path = "energy_distribution.png",
        title: str = "Energy Distribution"
) -> None:
    """ Plots the reconstructed energy distribution. """

    protons: list[float] = []
    electrons: list[float] = []
    for i, energy in enumerate(energies):
        if truths[i] == 0:
            protons.append(energy)
        else:
            electrons.append(energy)

    plt.hist(protons, bins='auto', histtype='step', label='Protons', alpha=0.7)
    plt.hist(electrons, bins='auto', histtype='step', label='Electrons', alpha=0.7)
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Reconstructed Energy [MeV]')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_error_energy_distribution(
        probs: torch.Tensor,
        truths: torch.Tensor,
        energies: NDArray[np.float32],
        save_path: str | Path = "error_energy_distribution.png",
        title: str = "Error vs Energy"
) -> None:
    """ Plots the confidence error against the event energy. """

    confidence = probs[torch.arange(len(truths)), truths].numpy()
    errors = 1.0 - confidence

    # 2. Filter invalid energies
    valid_mask = energies > 0
    clean_energies = energies[valid_mask]
    clean_errors = errors[valid_mask]
    clean_truths = truths.numpy()[valid_mask]

    # 3. Create bins (Y-axis now spans 0.0 to 1.0)
    x_bins = np.logspace(np.log10(clean_energies.min()), np.log10(clean_energies.max()), num=40)
    y_bins = np.linspace(0.0, 1.0, num=40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # --- Proton Error Distribution (truths == 0) ---
    p_mask = clean_truths == 0
    _, _, _, im1 = ax1.hist2d(
        clean_energies[p_mask], clean_errors[p_mask], 
        bins=[x_bins, y_bins], cmap='viridis', cmin=1, norm=LogNorm()
    )
    ax1.set_xscale('log')
    ax1.set_title("Protons: $1.0 - P(p)$")
    ax1.set_xlabel('Reconstructed Energy [MeV]')
    ax1.set_ylabel('Prediction Error')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    fig.colorbar(im1, ax=ax1, label='Events')

    # --- Electron Error Distribution (truths == 1) ---
    e_mask = clean_truths == 1
    _, _, _, im2 = ax2.hist2d(
        clean_energies[e_mask], clean_errors[e_mask], 
        bins=[x_bins, y_bins], cmap='plasma', cmin=1, norm=LogNorm()
    )
    ax2.set_xscale('log')
    ax2.set_title("Electrons: $1.0 - P(e)$")
    ax2.set_xlabel('Reconstructed Energy [MeV]')
    ax2.set_ylabel('Prediction Error')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    fig.colorbar(im2, ax=ax2, label='Events')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
