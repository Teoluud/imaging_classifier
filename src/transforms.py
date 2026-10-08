from typing import Protocol, Self
from pathlib import Path

import torch
import numpy as np


# index of the reconstructed energy [MeV] in `meta`
ENERGY_IDX = 2


class Transform(Protocol):
    """ 
    Protocol class for a generic transform.
    
    Maps one event's data (and the metadata row) to the tensor fed to the model.
    """

    def __call__(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor: ...


class Persistable(Protocol):
    """ A transform with fitted state that must be saved next to the model. """

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Self: ...


class ImageLogNormalizer:
    """ Normalize the log of the energy of the event display image w.r.t. the log of the event reconstructed energy. """

    def __call__(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)

        # Check for actual active pixels in the data tensor
        active = x > 0
        if active.any():
            # TKR mask -> shape (1, rows, 1), broadcasts across events and columns.
            tkr_mask = (torch.arange(x.shape[1], device=x.device) > 10)[None, :, None]
            # Multiplying factor: 1000 if in CAL, 5000 if in TKR for Mips -> fC conversion.
            factor = torch.where(tkr_mask, 5000.0, 1000.0).to(dtype=x.dtype)
            # Convert to keV without converting the tensor to a NumPy array.
            active_kev = torch.where(active, x * factor, x)
            event_energy_kev = float(meta[ENERGY_IDX]) * 1000.0
            log_norm_factor = np.log10(max(event_energy_kev, 1.0))  # to ensure positive normalization
            out[active] = torch.log10(active_kev) / log_norm_factor

        return out


class MeritMinMaxNormalizer:
    """ Normalizes the merit values using min max. """

    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor) -> None:
            self.min_value = min_value
            self.max_value = max_value
    
    @classmethod
    def fit(cls, data: torch.Tensor) -> Self:
        """ Computes the normalizing parameters for the dataset. """
        return cls(data.min(dim=0).values, data.max(dim=0).values)

    def __call__(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """ 
        Allows the instance to be called like a function.
        Returns the normalized merit tensor.

        Args:
            x (torch.Tensor): The merit variables data.
            meta (torch.Tensor): The meta data, not used (needed to comply with Transform protocol).
        """
        return (x - self.min_value) / (self.max_value - self.min_value)

    def save(self, path: Path) -> None:
        """ Saves the fitted state. """
        torch.save({"min": self.min_value, "max": self.max_value}, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """ Loads the fitted state from the save file. """
        state = torch.load(path, weights_only=False)
        return cls(state["min"], state["max"])