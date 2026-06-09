from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from logger import logger


class FermiLATDataset(Dataset):
    def __init__(self, proton_path: str | Path | None, electron_path: str | Path | None) -> None:
        """
        Loads the compressed numpy chunks into memory and assings labels:
        0 = Proton
        1 = Electron
        """
        x_list, y_list, top_list, meta_list, label_list = [], [], [], [], []

        # Load protons
        if proton_path is not None:
            logger.info(f"Loading Protons from {Path(proton_path).name}...")
            with np.load(proton_path) as archive:
                x_list.append(archive["view_x"])
                y_list.append(archive["view_y"])
                top_list.append(archive["view_top"])
                
                p_meta = archive["meta"]
                meta_list.append(p_meta)

                # Assign labels
                label_list.append(np.zeros(p_meta.shape[0], dtype=np.int64))
        else:
            logger.info("No Proton file selected...")
        
        # Load electrons
        if electron_path is not None:
            logger.info(f"Loading Electrons from {Path(electron_path).name}...")
            with np.load(electron_path) as archive:
                x_list.append(archive["view_x"])
                y_list.append(archive["view_y"])
                top_list.append(archive["view_top"])

                e_meta = archive["meta"]
                meta_list.append(e_meta)

                label_list.append(np.ones(e_meta.shape[0], dtype=np.int64))
        else:
            logger.info("No Electron file selected...")

        # Safety check
        if len(meta_list) == 0:
            raise ValueError("Both paths cannot be None! Please provide at least one dataset.")
        
        # Concatenate arrays along the batch dimension
        self.view_x = np.concatenate(x_list, axis=0)
        self.view_y = np.concatenate(y_list, axis=0)
        self.view_top = np.concatenate(top_list, axis=0)
        self.meta = np.concatenate(meta_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0)

        logger.info(f"Dataset ready: {len(self.labels)} total events loaded.")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the raw matrices
        x = self.view_x[idx]
        y = self.view_y[idx]
        top = self.view_top[idx]

        # Stack into a 3 channel numpy array (Shape: 3, 113, 113)
        stacked_views = np.stack([x, y, top], axis=0)

        # Convert to PyTorch tensor
        tensor_data = torch.from_numpy(stacked_views).type(torch.float)

        # We need a masked version to avoid taking the log(0)
        # Create an output tensor initialized with 0.0
        norm_tensor = torch.zeros_like(tensor_data)
        # Create a boolean mask
        active_pixels = tensor_data > 0

        # Apply normalization to active pixels
        if active_pixels.any():
            # Convert active pixels to keV
            active_kev = tensor_data[active_pixels] * 1000.0
            # Get the event energy
            event_energy_kev = self.meta[idx, 2] * 1000.0 # <- meta[:, 2] is the reconstructed energy
            log_norm_factor = np.log10(max(event_energy_kev, 1.0)) # <- Avoid normalizing to negative values
            # Normalize and overlay to norm_tensor
            norm_tensor[active_pixels] = torch.log10(active_kev) / log_norm_factor
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return norm_tensor, label


class ImportData:
    def __init__(self, proton_path: str | Path | None, electron_path: str | Path | None) -> None:
        self.dataset = FermiLATDataset(proton_path, electron_path)

        self.train_loader = None
        self.val_loader = None

    def train_split(self, split: float, random_state: int = 42) -> tuple[DataLoader, DataLoader]:
        """ Splits the data into train and validation DataLoaders.
        """
        self.generator = torch.Generator().manual_seed(random_state)

        train_size = int(split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset=self.dataset,
                                                                   lengths=[train_size, val_size],
                                                                   generator=self.generator)
        
        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=32,
                                       shuffle=True)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=32,
                                     shuffle=False)
        return self.train_loader, self.val_loader