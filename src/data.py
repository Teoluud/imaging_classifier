import bisect
from pathlib import Path
from typing import override, Any

# pyright: reportMissingTypeStubs=false
import numpy as np
from numpy.typing import NDArray
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset

from src.transforms import Transform, Persistable, ImageLogNormalizer, MeritMinMaxNormalizer
from src.logger import logger


class FermiLATDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Parent Dataset Class to load data from the .hdf5 files.
    Uses memory mapping and binary search to avoid filling the RAM and CPU bottlenecks.
    """

    def __init__(self, file_path: str | Path, transform: Transform | None = None) -> None:
        """ Constructor. """

        path = Path(file_path)
        assert(path.is_dir())
        self.proton_files = sorted(path.glob('protons/*.hdf5'))
        self.electron_files = sorted(path.glob('electrons/*.hdf5'))
        self.file_ranges: list[tuple[Path, int, int, int]] = []
        self.events_counter: int = 0
        # Store open file handles
        self.handles: dict[Path, h5py.File] = {}

        self.all_energies: list[NDArray[np.float32]] = []

        self._read_metadata()

        self.energies = np.concatenate(self.all_energies).astype(np.float32)
        del self.all_energies
        
        # Create a flat list of starting indices for fast binary search
        self.start_indices = [start_idx for _, start_idx, _, _ in self.file_ranges]

        self.transform = transform
        
        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")
        logger.info(f"Protons: {self._get_label_count(0)} | Electrons: {self._get_label_count(1)}")

    def _get_label_count(self, label: int) -> int:
        """ Returns the event count for the specified label. """

        count = 0
        for _, _, c, l in self.file_ranges:
            if l == label:
                count += c
        return count

    def _read_metadata(self) -> None:
        """ Parses chunk lenghts and assigns classification labels. """

        for path in self.proton_files:
            self._register_chunk(path, label=0)

        for path in self.electron_files:
            self._register_chunk(path, label=1)

    def _register_chunk(self, path: Path, label: int) -> None:
        """ Registers file ranges. """

        if not path.exists():
            raise FileExistsError(f"Chunk file not found: {path}")

        with h5py.File(path, "r") as f:
            node_meta = f["meta"]
            if isinstance(node_meta, h5py.Dataset):
                num_events: int = node_meta.shape[0]
                energy_values: NDArray[np.float32] = np.asarray(node_meta[:, 2], dtype=np.float32)
                self.all_energies.append(energy_values)
                self.file_ranges.append((path, self.events_counter, num_events, label))
                self.events_counter += num_events
            else:
                raise TypeError(f"Expected Dataset in {path}")

    def _get_handle(self, path: Path) -> h5py.File:
        """ Returns an open HDF5 file handle, opening it if necessary. """

        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", swmr=True)
        return self.handles[path]

    def _locate(self, idx: int) -> tuple[Path, int, int]:
        """ Finds the event in the data files. """
        # Binary search
        file_idx = bisect.bisect_right(self.start_indices, idx) - 1
        if file_idx < 0 or file_idx >= len(self.file_ranges):
            raise IndexError(f"Index {idx} out of bounds.")
        path, start, n, label = self.file_ranges[file_idx]
        if not (start <= idx < start + n):
            raise IndexError(f"Index {idx} out of bounds.")
        return path, idx - start, label

    def _read(self, f: h5py.File, local_idx: int) -> torch.Tensor:
        """ Gets the event from the file. """
        raise NotImplementedError("This method has to be implemented in a subclass.")
    
    @property
    def labels(self) -> np.ndarray:
        """ Reconstructs the full label array from chunk metadata for stratification. """

        return np.concatenate([
            np.full(num_events, label, dtype=np.int64)
            for _, _, num_events, label in self.file_ranges
        ])
    
    def __len__(self):
        return self.events_counter
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, local_idx, label = self._locate(idx)
        file = self._get_handle(path)
        data = self._read(file, local_idx)
        meta_node = file["meta"]
        assert isinstance(meta_node, h5py.Dataset)
        meta = torch.as_tensor(np.asarray(meta_node[local_idx], dtype=np.float64))
        if self.transform is not None:
            data = self.transform(data, meta)
        return data, torch.tensor(label, dtype=torch.long)

    def __del__(self) -> None:
        """ Closes all file handles when dataset is destroyed. """

        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def __getstate__(self) -> dict[str, Any]:
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
    """ Dataset Class to load imaging data (event display images). """

    @override
    def _read(self, f: h5py.File, local_idx: int) -> torch.Tensor:
        view_names = ("view_x", "view_y", "view_top")
        views: list[np.ndarray] = []
        for key in view_names:
            dataset = f[key]
            assert isinstance(dataset, h5py.Dataset)
            views.append(np.asarray(dataset[local_idx], dtype=np.float32))

        return torch.as_tensor(np.stack(views, axis=0))


class MeritDataset(FermiLATDataset):
    """ Dataset Class to load merit variables data. """

    @override
    def _read(self, f: h5py.File, local_idx: int) -> torch.Tensor:
        dataset = f["merit_values"]
        assert isinstance(dataset, h5py.Dataset)
        return torch.as_tensor(np.asarray(dataset[local_idx], dtype=np.float32))

    def load_all(self) -> torch.Tensor:
        """ Returns all merit vectors in global index order, shape (N, 17). """
        arrays: list[NDArray[np.float32]] = []
        for path, _, _, _ in self.file_ranges:
            with h5py.File(path, "r") as f:
                node = f["merit_values"]
                assert isinstance(node, h5py.Dataset)
                arrays.append(np.asarray(node[:], dtype=np.float32))
        return torch.as_tensor(np.concatenate(arrays))


class FermiDataModule:
    """ Manages training split and provides PyTorch DataLoaders. """

    def __init__(self, file_path: str | Path, batch_size: int = 32, merit: bool = False) -> None:
        self.merit = merit
        if merit:
            self.dataset = MeritDataset(file_path)
        else:
            self.dataset = ImagingDataset(file_path, transform=ImageLogNormalizer())
        self.batch_size = batch_size
        self.loaders: dict[str, DataLoader[Any]] = {}

    def setup_merit_normalizer(self, normalizer_cls: type[Persistable], save_path: Path, fit: bool) -> None:
        """ Fit on train split (or load a saved one) and attach to the dataset. """
        assert isinstance(self.dataset, MeritDataset)
        if fit:
            train_subset = self.loaders["train"].dataset
            assert isinstance(train_subset, Subset)
            data = self.dataset.load_all()[train_subset.indices]
            normalizer = normalizer_cls.fit(data)
            normalizer.save(save_path)
        else:
            normalizer = MeritMinMaxNormalizer.load(save_path)
        self.dataset.transform = normalizer

    def get_split_energies(self, split_name: str = "test") -> torch.Tensor:
        """ Returns the 1D energy tensor for the requested data split. """

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

        raise TypeError(f"Unsupported dataset type for split '{split_name}': {type(dataset)!r}")

    def train_split(
        self, train_split: float, test_split: float | None = None, random_state: int = 42
    ) -> dict[str, DataLoader[Any]]:
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
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=True)
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
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True)
        self.loaders["val"] = val_loader

        logger.info("TRAIN-TEST Split created.")
        return self.loaders