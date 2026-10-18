import bisect
from pathlib import Path
from typing import override, Any

# pyright: reportMissingTypeStubs=false
import numpy as np
from numpy.typing import NDArray
import torch
import h5py
from torch.utils.data import Dataset

from src.data.transforms import Transform
from src.logger import logger

N_CLASS_EVENTS = 200000
CHUNK_SIZE = 50000


class FermiLATDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Parent Dataset Class to load data from the .hdf5 files.
    Uses memory mapping and binary search to avoid filling the RAM and CPU bottlenecks.
    """

    def __init__(
        self, file_path: str | Path, transform: Transform | None = None
    ) -> None:
        """Constructor."""

        path = Path(file_path)
        assert path.is_dir()
        n_files = int(N_CLASS_EVENTS / CHUNK_SIZE)
        self.proton_files = sorted(path.glob("protons/*.hdf5"))[:n_files]
        self.electron_files = sorted(path.glob("electrons/*.hdf5"))[:n_files]
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
        logger.info(
            f"Protons: {self._get_label_count(0)} | Electrons: {self._get_label_count(1)}"
        )

    def _get_label_count(self, label: int) -> int:
        """Returns the event count for the specified label."""

        count = 0
        for _, _, c, l in self.file_ranges:
            if l == label:
                count += c
        return count

    def _read_metadata(self) -> None:
        """Parses chunk lenghts and assigns classification labels."""

        for path in self.proton_files:
            self._register_chunk(path, label=0)

        for path in self.electron_files:
            self._register_chunk(path, label=1)

    def _register_chunk(self, path: Path, label: int) -> None:
        """Registers file ranges."""

        if not path.exists():
            raise FileExistsError(f"Chunk file not found: {path}")

        with h5py.File(path, "r") as f:
            node_meta = f["meta"]
            if isinstance(node_meta, h5py.Dataset):
                num_events: int = node_meta.shape[0]
                energy_values: NDArray[np.float32] = np.asarray(
                    node_meta[:, 2], dtype=np.float32
                )
                self.all_energies.append(energy_values)
                self.file_ranges.append((path, self.events_counter, num_events, label))
                self.events_counter += num_events
            else:
                raise TypeError(f"Expected Dataset in {path}")

    def _get_handle(self, path: Path) -> h5py.File:
        """Returns an open HDF5 file handle, opening it if necessary."""

        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r", swmr=True)
        return self.handles[path]

    def _locate(self, idx: int) -> tuple[Path, int, int]:
        """Finds the event in the data files."""
        # Binary search
        file_idx = bisect.bisect_right(self.start_indices, idx) - 1
        if file_idx < 0 or file_idx >= len(self.file_ranges):
            raise IndexError(f"Index {idx} out of bounds.")
        path, start, n, label = self.file_ranges[file_idx]
        if not (start <= idx < start + n):
            raise IndexError(f"Index {idx} out of bounds.")
        return path, idx - start, label

    def _read(self, f: h5py.File, local_idx: int) -> torch.Tensor:
        """Gets the event from the file."""
        raise NotImplementedError("This method has to be implemented in a subclass.")

    @property
    def labels(self) -> np.ndarray:
        """Reconstructs the full label array from chunk metadata for stratification."""

        return np.concatenate(
            [
                np.full(num_events, label, dtype=np.int64)
                for _, _, num_events, label in self.file_ranges
            ]
        )

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
        """Closes all file handles when dataset is destroyed."""

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
        state["handles"] = {}
        return state


class ImagingDataset(FermiLATDataset):
    """Dataset Class to load imaging data (event display images)."""

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
    """Dataset Class to load merit variables data."""

    @override
    def _read(self, f: h5py.File, local_idx: int) -> torch.Tensor:
        dataset = f["merit_values"]
        assert isinstance(dataset, h5py.Dataset)
        return torch.as_tensor(np.asarray(dataset[local_idx], dtype=np.float32))

    def load_all(self) -> torch.Tensor:
        """Returns all merit vectors in global index order, shape (N, 17)."""
        arrays: list[NDArray[np.float32]] = []
        for path, _, _, _ in self.file_ranges:
            with h5py.File(path, "r") as f:
                node = f["merit_values"]
                assert isinstance(node, h5py.Dataset)
                arrays.append(np.asarray(node[:], dtype=np.float32))
        return torch.as_tensor(np.concatenate(arrays))
