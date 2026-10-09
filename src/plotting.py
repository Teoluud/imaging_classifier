from pathlib import Path
from typing import Any, cast

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import numpy as np
from numpy.typing import NDArray
import torch
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryROC
from mlxtend.plotting import plot_confusion_matrix


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


class EvaluationPlotter:
    """ Draws the evaluation plots of one model on one data split. """

    def __init__(self, plots_dir: Path, class_names: tuple[str, ...], model_name: str, split_name: str) -> None:
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.suffix = f"{model_name}, {split_name} dataset."

    def _finish(self, fig: Figure, filename: str, title: str) -> None:
        fig.suptitle(f"{title}: {self.suffix}")
        fig.savefig(self.plots_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def plot_conf_matrix(
            self,
            preds: torch.Tensor,
            truths: torch.Tensor,
            filename: str = "confusion_matrix.png",
            title: str = "Confusion Matrix"
    ) -> None:
        """ Generates and saves a multiclass confusion matrix. """
        confmat = ConfusionMatrix(task="multiclass", num_classes=len(self.class_names))
        confmat_tensor = confmat(preds=preds, target=truths)

        fig, _ = cast(
            tuple[Any, Any],
            plot_confusion_matrix(
                conf_mat=confmat_tensor.numpy(),
                class_names=self.class_names,
                figsize=(10, 7),
                show_normed=True,
            )
        )

        self._finish(fig, filename, title)

    def plot_roc_curve(
            self,
            probs: torch.Tensor,
            truths: torch.Tensor,
            filename: str = "roc_curve.png",
            title: str = "ROC Curve"
    ) -> None:
        """ Generates and saves a Binary ROC Curve, evaluating the positive class. """

        roc = BinaryROC()
        # probs[:, 1] extracts the probabilities for the positive class (Electron)
        roc.update(preds=probs[:, 1], target=truths)

        fig, _ = cast(tuple[Any, Any], roc.plot(score=True))
        self._finish(fig, filename, title)


    def plot_probs_distribution(
            self,
            probs: torch.Tensor,
            truths: torch.Tensor,
            filename: str = "probs_distribution.png",
            title: str = "Output probs distribution"
    ) -> None:
        """ Plots the distributions of the log(1 - probability), with the probability normalized between -1 (Proton) and 1 (Electron). """
        # 1 - (p_e - p_p) == 2 * p_p when p_e + p_p == 1; avoids catastrophic cancellation
        eps = torch.finfo(torch.float32).tiny
        norm_probs = torch.log((2.0 * probs[:, 0]).clamp(min=eps))

        protons = norm_probs[truths == 0].numpy()
        electrons = norm_probs[truths == 1].numpy()

        bins = np.linspace(norm_probs.min().item(), norm_probs.max().item(), 100)
        fig = plt.figure()
        plt.hist(protons, bins=bins.tolist(), histtype='step', label='Protons', alpha=0.7)
        plt.hist(electrons, bins=bins.tolist(), histtype='step', label='Electrons', alpha=0.7)
        plt.xlabel('log(1-p)')
        plt.yscale('log')
        plt.legend()
        self._finish(fig, filename, title)
        
    def plot_energy_distribution(
            self,
            energies: NDArray[np.float32],
            truths: torch.Tensor,
            filename: str = "energy_distribution.png",
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

        fig = plt.figure()
        plt.hist(protons, bins='auto', histtype='step', label='Protons', alpha=0.7)
        plt.hist(electrons, bins='auto', histtype='step', label='Electrons', alpha=0.7)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('Reconstructed Energy [MeV]')
        self._finish(fig, filename, title)


    def plot_error_energy_distribution(
            self,
            probs: torch.Tensor,
            truths: torch.Tensor,
            energies: NDArray[np.float32],
            filename: str = "error_energy_distribution.png",
            title: str = "Error vs Energy"
    ) -> None:
        """ Plots the confidence error against the event energy. """

        confidence = probs[torch.arange(len(truths)), truths].numpy()
        errors = 1.0 - confidence

        # Filter invalid energies
        valid_mask = energies > 0
        clean_energies = energies[valid_mask]
        clean_errors = errors[valid_mask]
        clean_truths = truths.numpy()[valid_mask]

        # Create bins (Y-axis spans 0.0 to 1.0)
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

        self._finish(fig, filename, title)
