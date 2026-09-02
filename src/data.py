import bisect
from pathlib import Path
from typing import List

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset

from src.logger import logger


class FermiLATDataset(Dataset):
    """
    Dataset to load data from the .hdf5 files.
    Uses memory mapping and binary search to avoid filling the RAM and CPU bottlenecks.
    """

    def __init__(self, file_path: str | Path) -> None:
        """ Constructor.
        """
        path = Path(file_path)
        assert(path.is_dir())
        self.proton_files = sorted(path.glob('protons/*.hdf5'))
        self.electron_files = sorted(path.glob('electrons'))
        self.file_ranges = []
        self.events_counter = 0
        # Store open file handles
        self.handles: dict[Path, h5py.File] = {}
        
        self._read_metadata()
        
        # Create a flat list of starting indices for fast binary search
        self.start_indices = [start_idx for _, start_idx, _, _ in self.file_ranges]
        
        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

    def _read_metadata(self) -> None:
        """ Parses chunk lenghts and assigns classification labels.
        """
        for path in self.proton_files:
            self._register_chunk(path, label=0)

        for path in self.electron_files:
            self._register_chunk(path, label=1)

    def _register_chunk(self, path: Path, label: int) -> None:
        """ Registers file ranges.
        """
        if not path.exists():
            raise FileExistsError(f"Chunk file not found: {path}")

        with h5py.File(path, "r") as f:
            node_meta = f["meta"]
            if isinstance(node_meta, h5py.Dataset):
                num_events = node_meta.shape[0]
                self.file_ranges.append((path, self.events_counter, num_events, label))
                self.events_counter += num_events
            else:
                raise TypeError(f"Expected Dataset in {path}")

    def _get_handle(self, path: Path) -> h5py.File:
        """ Returns an open HDF5 file handle, opening it if necessary.
        """
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", swmr=True)
        return self.handles[path]

    @property
    def labels(self) -> np.ndarray:
        """Reconstructs the full label array from chunk metadata for stratification."""
        return np.concatenate([
            np.full(num_events, label, dtype=np.int64)
            for _, _, num_events, label in self.file_ranges
        ])
    
    def __len__(self):
        return self.events_counter
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Binary search instantly finds the correct chunk
        file_idx = bisect.bisect_right(self.start_indices, idx) - 1
        
        if file_idx < 0 or file_idx >= len(self.file_ranges):
            raise IndexError(f"Index {idx} out of bounds.")
            
        target_path, start_idx, num_events, event_label = self.file_ranges[file_idx]
        
        # Safety check to ensure the index is valid for this chunk
        if not (start_idx <= idx < start_idx + num_events):
            raise IndexError(f"Index {idx} out of bounds.")
            
        local_idx = idx - start_idx

        # Retrieve file handle
        f = self._get_handle(target_path)
        
        # Extract data
        # Type checking
        node_x = f["view_x"]
        node_y = f["view_y"]
        node_top = f["view_top"]
        node_meta = f["meta"]
        
        if (isinstance(node_x, h5py.Dataset) and 
            isinstance(node_y, h5py.Dataset) and 
            isinstance(node_top, h5py.Dataset) and 
            isinstance(node_meta, h5py.Dataset)):
            x = node_x[local_idx]
            y = node_y[local_idx]
            top = node_top[local_idx]
            event_meta = node_meta[local_idx]
        else:
            raise TypeError("Expected h5py.Dataset")

        # Stack into a 3-channel numpy array (Shape: 3, 113, 113)
        stacked_views = np.stack([x, y, top], axis=0)
        tensor_data = torch.from_numpy(stacked_views).type(torch.float)

        # Log-normalization
        norm_tensor = torch.zeros_like(tensor_data)
        active_pixels = tensor_data > 0

        if active_pixels.any():
            active_kev = tensor_data[active_pixels] * 1000.0
            event_energy_kev = event_meta[2] * 1000.0
            log_norm_factor = np.log10(max(event_energy_kev, 1.0))
            norm_tensor[active_pixels] = torch.log10(active_kev) / log_norm_factor
        
        # Get label
        label = torch.tensor(event_label, dtype=torch.long)
        
        return norm_tensor, label

    def __del__(self) -> None:
        """ Closes all file handles when dataset is destroyed.
        """
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass


class FermiMeritDataset(Dataset):
    """ Dataset wrapper for events merit variables.
    """

    def __init__(self, proton_path: str | Path | None, electron_path: str | Path | None) -> None:
        """
        Loads the compressed numpy chunks and assings classification labels.

        Labels:
            0 = Proton
            1 = Electron
        """
        merit_vars, meta_list, label_list = [], [], []

        # Load Protons
        if proton_path is not None:
            logger.debug(f"Loading Protons from {Path(proton_path).name}...")
            with np.load(proton_path) as archive:
                merit_vars.append(archive["merit_values"])

                p_meta = archive["meta"]
                meta_list.append(p_meta)
                label_list.append(np.zeros(p_meta.shape[0], dtype=np.int64))
        else:
            logger.info("No Proton file selected...")
        
        if electron_path is not None:
            logger.debug(f"Loading Electrons from {Path(electron_path).name}...")
            with np.load(electron_path) as archive:
                merit_vars.append(archive["merit_values"])

                e_meta = archive["meta"]
                meta_list.append(e_meta)
                label_list.append(np.ones(e_meta.shape[0], dtype=np.int64))
        else:
            logger.info("No Electron file selected...")

        # Safety check
        if len(meta_list) == 0:
            raise ValueError("Both paths cannot be None! Please provide at least one dataset.")

        self.raw_merit_vars = np.concatenate(merit_vars, axis=0)

        self.meta = np.concatenate(meta_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0)

        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        merit_var = self.raw_merit_vars[idx]
        label = self.labels[idx]

        return torch.from_numpy(merit_var).type(torch.float), torch.tensor(label, dtype=torch.long)
        

class FermiDataModule:
    """ Manages training split and provides PyTorch DataLoaders.
    """

    def __init__(
            self,
            file_path: str | Path,
            batch_size: int = 32,
            merit: bool = False
    ) -> None:
        if merit:
            self.dataset = FermiMeritDataset(file_path)
        else:
            self.dataset = FermiLATDataset(file_path)
        self.batch_size = batch_size
        self.train_loader = None
        self.val_loader = None

    def train_split(self, split: float, random_state: int = 42) -> tuple[DataLoader, DataLoader]:
        """ Splits the data into train and validation DataLoaders.
        """
        np.random.seed(random_state)
        
        labels = self.dataset.labels
        # Isolate indices by particle type
        proton_idx = np.where(labels == 0)[0]
        electron_idx = np.where(labels == 1)[0]
        
        np.random.shuffle(proton_idx)
        np.random.shuffle(electron_idx)
        
        # Calculate split limits
        p_split = int(len(proton_idx) * split)
        e_split = int(len(electron_idx) * split)

        train_indices = np.concatenate((proton_idx[:p_split], electron_idx[:e_split]))
        val_indices = np.concatenate((proton_idx[p_split:], electron_idx[e_split:]))

        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        train_dataset = Subset(self.dataset, train_indices.tolist())
        val_dataset = Subset(self.dataset, val_indices.tolist())
        
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)
        logger.debug("TRAIN-TEST Split created.")
        return self.train_loader, self.val_loader
    
    def get_test_dataset(
        self,
        file_path: str | Path,
        merit: bool = False
    ) -> DataLoader:
        """ Creates an optimized DataLoader for evaluation on unseen datasets. """
        if merit:
            test_dataset = FermiMeritDataset(proton_files, electron_files)
        else:
            test_dataset = FermiLATDataset(file_path)

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size * 2,  # Double batch size for inference
            shuffle=False,
            num_workers=8,                   # Parallelize HDF5 reads
            pin_memory=True                  # Speed up CPU to GPU transfer
        )