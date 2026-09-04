import bisect
from pathlib import Path
from typing import override, Callable

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils import normalize_image, normalize_merit
from src.logger import logger


class FermiLATDataset(Dataset):
    """
    Parent Dataset Class to load data from the .hdf5 files.
    Uses memory mapping and binary search to avoid filling the RAM and CPU bottlenecks.
    """

    def __init__(self, file_path: str | Path) -> None:
        """ Constructor.
        """
        path = Path(file_path)
        assert(path.is_dir())
        self.proton_files = sorted(path.glob('protons/*.hdf5'))
        self.electron_files = sorted(path.glob('electrons/*.hdf5'))
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
        raise NotImplementedError("This method has to be implemented in a subclass.")

    def __del__(self) -> None:
        """ Closes all file handles when dataset is destroyed.
        """
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def __getstate__(self) -> dict:
        """
        Prevents PyTorch multiprocessing pickling errors.
        Strips the unpicklable open C-level file handles from the object state
        before sending a copy of the dataset to the spawned child workers.
        """
        state = self.__dict__.copy()
        # Wipe the handles dictionary for the child workers
        state['handles'] = {}
        return state


class ImagingDataset(FermiLATDataset):
    """ Dataset Class to load imaging data (event display images).
    """

    def __init__(self, file_path: str | Path, transform: Callable[..., torch.Tensor] | None = None) -> None:
        super().__init__(file_path)
        self.transform = transform

    @override
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

        if self.transform is not None:
            tensor_data = self.transform(tensor_data, event_meta[2])
        
        # Get label
        label = torch.tensor(event_label, dtype=torch.long)
        
        return tensor_data, label


class MeritDataset(FermiLATDataset):
    """ Dataset Class to load merit variables data.
    """

    def __init__(self, file_path: str | Path, transform: Callable[..., torch.Tensor] | None = None) -> None:
        super().__init__(file_path)
        self.transform = transform

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Search for the file idx
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
        node_var = f["merit_values"]
        node_meta = f["meta"]

        if isinstance(node_var, h5py.Dataset) and isinstance(node_meta, h5py.Dataset):
            merit_var = node_var[local_idx]
            event_meta = node_meta[local_idx]
        else:
            raise TypeError("Expected h5py.Dataset")
        merit_var = torch.from_numpy(merit_var).type(torch.float)

        if self.transform is not None:
            merit_var = self.transform(merit_var)

        return merit_var, torch.tensor(event_label, dtype=torch.long)
        

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
            self.dataset = MeritDataset(file_path, transform=normalize_merit)
        else:
            self.dataset = ImagingDataset(file_path, transform=normalize_image)
        self.batch_size = batch_size
        self.loaders: dict[str, DataLoader] = {}

    def train_split(
        self, train_split: float, test_split: float | None = None, random_state: int = 42
    ) -> dict[str, DataLoader]:
        """ Splits the data into train and validation DataLoaders.
        """
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
        train_indices = np.concatenate((proton_idx[:p_train_split], electron_idx[:e_train_split]))
        np.random.shuffle(train_indices)
        train_dataset = Subset(self.dataset, train_indices.tolist())
        # Create DataLoaders
        train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.loaders["train"] = train_loader

        # VALIDATION AND OPTIONAL TEST SPLIT
        if test_split is not None:
            p_test_split = int(len(proton_idx) * (train_split + test_split))
            e_test_split = int(len(electron_idx) * (train_split + test_split))
            test_indices = np.concatenate((proton_idx[p_test_split:], electron_idx[e_test_split:]))
            np.random.shuffle(test_indices)
            test_dataset = Subset(self.dataset, test_indices.tolist())
            test_loader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=8,    # Parallelize HDF5 reads
                                          pin_memory=True)  # Speed up CPU to GPU transfer
            self.loaders["test"] = test_loader
            
            val_indices = np.concatenate((proton_idx[p_train_split:p_test_split], electron_idx[e_train_split:e_test_split]))

        else:
            val_indices = np.concatenate((proton_idx[p_train_split:], electron_idx[e_train_split:]))

        np.random.shuffle(val_indices)
        val_dataset = Subset(self.dataset, val_indices.tolist())
        val_loader = DataLoader(val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)
        self.loaders["val"] = val_loader

        logger.debug("TRAIN-TEST Split created.")
        return self.loaders