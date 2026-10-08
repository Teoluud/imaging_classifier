from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import torch
from torchmetrics.classification import MulticlassAccuracy


class Labels(Enum):
    PROTON = 0
    ELECTRON = 1


@dataclass(kw_only=True)
class Config:
    """ General configuration settings.
    """
    class_names: tuple[str, ...] = tuple(Labels.__members__.keys())

    # Data Paths
    data_dir: Path = Path("/data1/grosjacques/data/unpacked")
    test_data_dir: Path = Path("/data1/grosjacques/data/unpacked")

    # Output Path
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    model_save_path: Path
    plots_dir: Path = output_dir / "plots"

    def _plot_path(self, filename: str) -> Path:
        return self.plots_dir / filename

    @property
    def loss_plot_save_path(self) -> Path:
        return self._plot_path("loss_curves.png")

    @property
    def conf_matrix_save_path(self) -> Path:
        return self._plot_path("confusion_matrix.png")

    @property
    def roc_curve_save_path(self) -> Path:
        return self._plot_path("roc_curve.png")

    @property
    def probs_distribution_save_path(self) -> Path:
        return self._plot_path("probs_distribution.png")

    @property
    def error_energy_distribution_save_path(self) -> Path:
        return self._plot_path("error_energy_distribution.png")

    # Hyperparameters
    random_seed: int = 42
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 64
    train_split: float = 0.8
    test_split: float | None = 0.10
    epochs: int = 50

    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()
    accuracy_fn: MulticlassAccuracy = MulticlassAccuracy(num_classes=2, average="micro")


@dataclass(kw_only=True)
class MultiBranchConfig(Config):
    """ Configuration settings specific to multibranch CNN model.
    """
    model_save_path: Path = Config.output_dir / "models" / "multi_branch_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "multibranch"


@dataclass(kw_only=True)
class SingleBranchConfig(Config):
    """ Configuration settings specific to singlebranch CNN model.
    """
    model_save_path: Path = Config.output_dir / "models" / "single_branch_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "singlebranch"


@dataclass(kw_only=True)
class MeritConfig(Config):
    """ Configuration settings specific to merit variables model.
    """
    model_save_path: Path = Config.output_dir / "models" / "merit_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "merit"
    normalizer_save_path: Path = Config.output_dir / "models" / "merit_normalizer.pth"

    # Hyperparameters
    weight_decay: float = 0.
    learning_rate: float = 0.001