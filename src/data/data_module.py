import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import FermiLATDataset, MeritDataset
from src.data.transforms import NormalizerSettings
from src.logger import logger

type Loader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


class FermiDataModule:
    """Manages training split and provides PyTorch DataLoaders."""

    def __init__(self, dataset: FermiLATDataset, batch_size: int = 32) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.loaders: dict[str, Loader] = {}

    def setup_merit_normalizer(
        self, normalizer:NormalizerSettings, fit: bool
    ) -> None:
        """Fit on train split (or load a saved one) and attach to the dataset."""
        if not isinstance(self.dataset, MeritDataset):
            raise TypeError(f"Can't setup Merit Normalizer for {(self.dataset)!r}")
        if fit:
            train_subset = self.loaders["train"].dataset
            assert isinstance(train_subset, Subset)
            data = self.dataset.load_all()[train_subset.indices]
            fitted = normalizer.cls.fit(data)
            fitted.save(normalizer.save_path)
        else:
            fitted = normalizer.cls.load(normalizer.save_path)
        self.dataset.transform = fitted

    def get_split_energies(self, split_name: str = "test") -> torch.Tensor:
        """Returns the 1D energy tensor for the requested data split."""

        loader = self.loaders[split_name]
        dataset = loader.dataset

        if isinstance(dataset, Subset):
            subset_indices = dataset.indices
            parent_dataset = dataset.dataset
            if isinstance(parent_dataset, FermiLATDataset):
                energies = parent_dataset.energies[subset_indices]
                return torch.as_tensor(energies, dtype=torch.float32)

        if isinstance(dataset, FermiLATDataset):
            return torch.as_tensor(dataset.energies, dtype=torch.float32)

        raise TypeError(
            f"Unsupported dataset type for split '{split_name}': {type(dataset)!r}"
        )

    def full_loader(self, shuffle: bool = False) -> Loader:
        """Returns the DataLoader for the full dataset."""
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
        )
        return loader

    def train_split(
        self, train_split: float, val_split: float | None = None, random_state: int = 42
    ) -> dict[str, Loader]:
        """Splits the data into train and validation DataLoaders."""
        np.random.seed(random_state)

        labels = self.dataset.labels
        # Isolate indices by particle type
        proton_idx = np.where(labels == 0)[0]
        electron_idx = np.where(labels == 1)[0]

        np.random.shuffle(proton_idx)
        np.random.shuffle(electron_idx)

        # TRAIN SPLIT
        p_train_split = int(len(proton_idx) * train_split)
        e_train_split = int(len(electron_idx) * train_split)
        train_indices = np.concatenate(
            (proton_idx[:p_train_split], electron_idx[:e_train_split])
        )
        np.random.shuffle(train_indices)
        train_dataset = Subset(self.dataset, train_indices.tolist())
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        self.loaders["train"] = train_loader

        # VALIDATION AND OPTIONAL TEST SPLIT
        if val_split is not None:
            p_val_split = int(len(proton_idx) * (train_split + val_split))
            e_val_split = int(len(electron_idx) * (train_split + val_split))
            test_indices = np.concatenate(
                (proton_idx[p_val_split:], electron_idx[e_val_split:])
            )
            np.random.shuffle(test_indices)
            test_dataset = Subset(self.dataset, test_indices.tolist())
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,  # Parallelize HDF5 reads
                pin_memory=True,
            )  # Speed up CPU to GPU transfer
            self.loaders["test"] = test_loader

            val_indices = np.concatenate(
                (
                    proton_idx[p_train_split:p_val_split],
                    electron_idx[e_train_split:e_val_split],
                )
            )

        else:
            val_indices = np.concatenate(
                (proton_idx[p_train_split:], electron_idx[e_train_split:])
            )

        np.random.shuffle(val_indices)
        val_dataset = Subset(self.dataset, val_indices.tolist())
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        self.loaders["val"] = val_loader

        logger.info("TRAIN-TEST Split created.")
        return self.loaders
