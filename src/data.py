from pathlib import Path
from typing import List, Callable

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.logger import logger


class FermiLATDataset(Dataset):
    """
    Dataset to load data from the .hdf5 files.
    Uses memory mapping to avoid filling the RAM.
    """

    def __init__(self, proton_files: List[Path], electron_files: List[Path]) -> None:
        """ Constructor.
        """
        self.proton_paths = [Path(p) for p in proton_files]
        self.electron_paths = [Path(p) for p in electron_files]
        self.file_ranges = []
        self.events_counter = 0
        # Store open file handles
        self.handles: dict[Path, h5py.File] = {}
        self._read_metadata()
        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

    def _read_metadata(self) -> None:
        """ Parses chunk lenghts and assigns classification labels.
        """
        for path in self.proton_paths:
            self._register_chunk(path, label=0)

        for path in self.electron_paths:
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
            self.handles[path] = h5py.File(path, "r")
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
        target_path = None
        local_idx = -1
        event_label = -1

        for path, start_idx, num_events, label in self.file_ranges:
            if start_idx <= idx < (start_idx + num_events):
                target_path = path
                local_idx = idx - start_idx
                event_label = label
                break
        
        if target_path is None:
            raise IndexError(f"Index {idx} out of bounds.")

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
    
        

# class FermiLATDataset(Dataset):
#     """ Dataset wrapper to load and log-normalize events.
#     """

#     def __init__(self, proton_path: str | Path | None, electron_path: str | Path | None) -> None:
#         """
#         Loads the compressed numpy chunks and assings classification labels.

#         Labels:
#             0 = Proton
#             1 = Electron
#         """
#         x_list, y_list, top_list, meta_list, label_list = [], [], [], [], []

#         # Load protons
#         if proton_path is not None:
#             logger.debug(f"Loading Protons from {Path(proton_path).name}...")
#             with np.load(proton_path) as archive:
#                 x_list.append(archive["view_x"])
#                 y_list.append(archive["view_y"])
#                 top_list.append(archive["view_top"])
                
#                 p_meta = archive["meta"]
#                 meta_list.append(p_meta)

#                 # Assign labels
#                 label_list.append(np.zeros(p_meta.shape[0], dtype=np.int64))
#         else:
#             logger.info("No Proton file selected...")
        
#         # Load electrons
#         if electron_path is not None:
#             logger.debug(f"Loading Electrons from {Path(electron_path).name}...")
#             with np.load(electron_path) as archive:
#                 x_list.append(archive["view_x"])
#                 y_list.append(archive["view_y"])
#                 top_list.append(archive["view_top"])

#                 e_meta = archive["meta"]
#                 meta_list.append(e_meta)

#                 label_list.append(np.ones(e_meta.shape[0], dtype=np.int64))
#         else:
#             logger.info("No Electron file selected...")

#         # Safety check
#         if len(meta_list) == 0:
#             raise ValueError("Both paths cannot be None! Please provide at least one dataset.")
        
#         # Concatenate arrays along the batch dimension
#         self.view_x = np.concatenate(x_list, axis=0)
#         self.view_y = np.concatenate(y_list, axis=0)
#         self.view_top = np.concatenate(top_list, axis=0)
#         self.meta = np.concatenate(meta_list, axis=0)
#         self.labels = np.concatenate(label_list, axis=0)

#         logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
#         # Get the raw matrices
#         x = self.view_x[idx]
#         y = self.view_y[idx]
#         top = self.view_top[idx]

#         # Stack into a 3 channel numpy array (Shape: 3, 113, 113)
#         stacked_views = np.stack([x, y, top], axis=0)

#         # Convert to PyTorch tensor
#         tensor_data = torch.from_numpy(stacked_views).type(torch.float)

#         # We need a masked version to avoid taking the log(0)
#         # Create an output tensor initialized with 0.0
#         norm_tensor = torch.zeros_like(tensor_data)
#         # Create a boolean mask
#         active_pixels = tensor_data > 0

#         # Apply normalization to active pixels
#         if active_pixels.any():
#             # Convert active pixels to keV
#             active_kev = tensor_data[active_pixels] * 1000.0
#             # Get the event energy
#             event_energy_kev = self.meta[idx, 2] * 1000.0 # <- meta[:, 2] is the reconstructed energy
#             log_norm_factor = np.log10(max(event_energy_kev, 1.0)) # <- Avoid normalizing to negative values
#             # Normalize and overlay to norm_tensor
#             norm_tensor[active_pixels] = torch.log10(active_kev) / log_norm_factor
        
#         # Get label
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
        
#         return norm_tensor, label


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

        # Assign normalize variables using their Z-score
        # clean_merit_vars = np.nan_to_num(raw_merit_vars, nan=0.0, posinf=0.0, neginf=0.0)
        # means = clean_merit_vars.mean(axis=0, keepdims=True)
        # stds = clean_merit_vars.std(axis=0, keepdims=True)
        # stds[stds == 0] = 1.0
        # self.merit_vars = (clean_merit_vars - means) / stds

        self.meta = np.concatenate(meta_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0)

        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # merit_var = self.merit_vars[idx]
        merit_var = self.raw_merit_vars[idx]
        label = self.labels[idx]

        return torch.from_numpy(merit_var).type(torch.float), torch.tensor(label, dtype=torch.long)
        

class FermiDataModule:
    """ Manages training split and provides PyTorch DataLoaders.
    """

    def __init__(
            self,
            proton_files: List[Path],
            electron_files: List[Path],
            batch_size: int = 32,
            merit: bool = False
    ) -> None:
        if merit:
            self.dataset = FermiMeritDataset(proton_files, electron_files)
        else:
            self.dataset = FermiLATDataset(proton_files, electron_files)
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
        electron_idx = np.where(labels == 0)[0]
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
        
        for inputs, labels in self.train_loader:
            logger.debug(f"Batch Inputs Shape: {inputs.shape}")
            logger.debug(f"Batch Labels Shape: {labels.shape}")
            logger.debug(f"Test: {inputs.shape}")
            break # <- Test the first batch
        return self.train_loader, self.val_loader
    
    def get_test_dataset(
            self,
            proton_files: List[Path],
            electron_files: List[Path],
            merit: bool = False
    ) -> DataLoader:
        """ Creates a DataLoader for evaluation on unseen datasets.
        """
        if merit:
            test_dataset = FermiMeritDataset(proton_files, electron_files)
        else:
            test_dataset = FermiLATDataset(proton_files, electron_files)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)