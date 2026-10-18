from dataclasses import dataclass, field, replace
from pathlib import Path
from enum import Enum

import torch
from torchmetrics.classification import MulticlassAccuracy

from src.data.datasets import FermiLATDataset, ImagingDataset, MeritDataset
from src.data.transforms import (
    Transform,
    NormalizerSettings,
    ImageLogNormalizer,
    MeritMinMaxNormalizer,
)


class Labels(Enum):
    PROTON = 0
    ELECTRON = 1


@dataclass(kw_only=True)
class HyperParameters:
    random_seed: int
    batch_size: int
    train_split: float
    val_split: float | None
    learning_rate: float
    weight_decay: float
    epochs: int


def default_hyper_params() -> HyperParameters:
    """Builds a fresh set of default hyperparameters (one per config instance)."""
    return HyperParameters(
        random_seed=42,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=64,
        train_split=0.8,
        val_split=0.10,
        epochs=50,
    )


@dataclass(kw_only=True)
class Config:
    """General configuration settings."""

    dataset_cls: type[FermiLATDataset]
    transform: Transform | None = None
    normalizer: NormalizerSettings | None = None

    class_names: tuple[str, ...] = tuple(Labels.__members__.keys())

    # Data Paths
    data_dir: Path = Path("/data1/grosjacques/data/30_200_GeV")
    evaluate_data_dir: Path = Path("/data1/grosjacques/data/unpacked")

    # Output Path
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    model_save_path: Path
    plots_dir: Path = output_dir / "plots"

    # Hyperparameters
    hyper_params: HyperParameters = field(default_factory=default_hyper_params)

    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()
    accuracy_fn: MulticlassAccuracy = MulticlassAccuracy(num_classes=2, average="micro")

    def _plot_path(self, filename: str) -> Path:
        return self.plots_dir / filename

    @property
    def loss_plot_save_path(self) -> Path:
        return self._plot_path("loss_curves.png")


@dataclass(kw_only=True)
class MultiBranchConfig(Config):
    """Configuration settings specific to multibranch CNN model."""

    dataset_cls: type[FermiLATDataset] = ImagingDataset
    transform: Transform | None = ImageLogNormalizer()

    model_save_path: Path = Config.output_dir / "models" / "multi_branch_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "multibranch"


@dataclass(kw_only=True)
class SingleBranchConfig(Config):
    """Configuration settings specific to singlebranch CNN model."""

    dataset_cls: type[FermiLATDataset] = ImagingDataset
    transform: Transform | None = ImageLogNormalizer()

    model_save_path: Path = Config.output_dir / "models" / "single_branch_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "singlebranch"


@dataclass(kw_only=True)
class MeritConfig(Config):
    """Configuration settings specific to merit variables model."""

    dataset_cls: type[FermiLATDataset] = MeritDataset
    normalizer: NormalizerSettings | None = NormalizerSettings(
        MeritMinMaxNormalizer, Config.output_dir / "models" / "merit_normalizer.pth"
    )

    model_save_path: Path = Config.output_dir / "models" / "merit_model.pth"
    plots_dir: Path = Config.output_dir / "plots" / "merit"

    # Hyperparameters
    hyper_params: HyperParameters = field(
        default_factory=lambda: replace(default_hyper_params(), weight_decay=0.0)
    )
