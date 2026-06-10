from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """ Configuration settings for the classification pipeline.
    """

    # Paths
    data_dir: Path = Path("/data1/grosjacques/data")
    proton_path: Path = data_dir / "dataset_chunk_allpro_no_norm.npz"
    electron_path: Path = data_dir / "dataset_chunk_allhee_no_norm.npz"
    test_electron_path: Path = data_dir / "dataset_chunk_electronSimo_no_norm.npz"

    output_dir = Path("outputs")
    model_save_path: Path = output_dir / "models" / "best_fermi_model.pth"
    plot_save_path: Path = output_dir/ "plots" / "loss_curves.png"
    conf_matrix_save_path: Path = output_dir / "plots" / "confusion_matrix.png"
    roc_curve_save_path: Path = output_dir / "plots" / "roc_curve.png"

    # Hyperparameters
    random_seed: int = 42
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    train_split: float = 0.8
    epochs: int = 50

    class_names: tuple[str, str] = ("Proton", "Electron")