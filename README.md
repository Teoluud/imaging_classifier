# Imaging classifier for Fermi-LAT detector events

PyTorch classifier that separates electron and proton events in the Fermi-LAT detector. Two CNNs work on the event display images (three views per event), and a small fully connected network works on the reconstructed (merit) variables as a more traditional baseline to compare against.

| Flag | Model | Input |
|------|-------|-------|
| `--single-branch` | `FermiSingleBranchCNN` | 3 views stacked as channels, `(3, 113, 113)` |
| `--multi-branch` | `FermiMultiBranchCNN` | same input, one conv branch per view, features concatenated |
| `--merit` | `FermiMeritVarsNN` | 17 merit variables |

All models output two logits `[proton, electron]` and are trained with cross-entropy.

## Repository layout

```
main.py              CLI entry point
src/
  config.py          Hyperparameters, paths, one config class per model
  data.py            HDF5 datasets, stratified split, DataLoaders
  transforms.py      Transform protocol, image and merit normalization
  model.py           CNN and MLP architectures
  training_loop.py   Train/validation loop, best-model checkpointing
  evaluator.py       Loss/accuracy and prediction collection on the test split
  pipelines.py       ClassifierPipeline: data -> (train) -> evaluate -> plots
  logger.py          tqdm-safe logger
  utils.py           Plotting helpers
outputs/             Generated plots and weights
```

## Data

`data_dir` (in `config.py`) must contain one HDF5 file per chunk:

```
<data_dir>/protons/*.hdf5      -> label 0
<data_dir>/electrons/*.hdf5    -> label 1
```

Each file holds the datasets `view_x`, `view_y`, `view_top` (images), `merit_values` (17 variables per event) and `meta` (column 2 is the reconstructed energy in MeV).

Datasets are never loaded into RAM. `FermiLATDataset` records the event range of every file, finds the file of a given index with a binary search, and reads the single event from an `h5py` handle that is opened lazily in each worker. Handles are dropped when the dataset is pickled, and `main.py` forces the `spawn` start method so HDF5 state is never shared between DataLoader workers.

`ImagingDataset` and `MeritDataset` only implement `_read`, which returns the raw tensor for one event. The base class reads the `meta` row, applies the transform and attaches the label.

## Transforms

Every transform follows the `Transform` protocol in `src/transforms.py`:

```python
def __call__(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor: ...
```

`x` is the event data and `meta` is the event's `meta` row, so image and merit datasets share the same code path. Transforms that need to be fitted also satisfy the `Persistable` protocol (`save(path)` / `load(path)`).

* **`ImageLogNormalizer`** (stateless). Pixels with a positive value are converted to keV (factor 1000 for the first 11 rows, which are the calorimeter part, and 5000 for the remaining rows, the tracker part), then `log10` is taken and divided by `log10` of the event's reconstructed energy in keV (floored at 1 keV). Empty pixels stay 0.
* **`MeritMinMaxNormalizer`** (fitted). Per-variable min-max scaling. The minimum and maximum are computed **on the training split only** and saved with `torch.save`.

Merit normalization is set up by `FermiDataModule.setup_merit_normalizer` after the split is created:

* with `--train` the normalizer is fitted on the training indices and saved;
* without `--train` the saved normalizer is loaded, so evaluation always uses the statistics the weights were trained with.

To add a transform, write any callable with the `(x, meta)` signature and pass it to the dataset. Add `save`/`load` if it has fitted state.

## Setup

Requirements: the code is run on Python 3.14.5, and uses `torch`, `numpy`, `h5py`, `matplotlib`, `tqdm`, `torchmetrics`, `mlxtend`, `argcomplete`.

```bash
conda create -n imaging_classifier python=3.12
conda activate imaging_classifier
# install PyTorch following pytorch.org for your CUDA version, then:
conda install numpy h5py matplotlib tqdm torchmetrics mlxtend argcomplete
```

## Usage

```
python main.py (--merit | --multi-branch | --single-branch) [--train] [--epochs EPOCHS] [-v]
```

| Option | Effect |
|--------|--------|
| `--merit`, `--multi-branch`, `--single-branch` | Choose the model (exactly one is required) |
| `--train` | Train a new model from scratch. Without it, the saved weights are loaded |
| `--epochs N` | Override `epochs` from `config.py` |
| `-v`, `--verbose` | Debug logging |

```bash
python main.py --single-branch --train --epochs 50   # train, then evaluate
python main.py --single-branch                       # evaluate the saved weights
python main.py --merit --train                       # fit the normalizer, train, evaluate
```

Evaluation always runs at the end, using the checkpoint with the lowest validation loss.

## Pipeline

1. **Split.** Protons and electrons are shuffled separately (seed `random_seed`) and divided into train/validation/test, so class proportions are preserved. Defaults: 80% / 10% / 10%. If `test_split` is `None`, the remaining 20% is used as the validation set and evaluation runs on it.
2. **Training** (`--train`). Adam with `learning_rate` and `weight_decay`, cross-entropy loss, `epochs` epochs. The weights with the lowest validation loss are saved to `model_save_path`.
3. **Evaluation.** The saved weights are loaded and run over the test split, then the plots below are produced.

## Configuration

Edit `src/config.py`. `Config` holds the shared defaults: `random_seed=42`, `learning_rate=1e-3`, `weight_decay=1e-4`, `batch_size=64`, `train_split=0.8`, `test_split=0.10`, `epochs=50`, and the data paths. `MultiBranchConfig`, `SingleBranchConfig` and `MeritConfig` only set the weights path and the plot directory (`MeritConfig` also sets `weight_decay=0`).

## Outputs

Weights go to `outputs/models/` (not tracked): `single_branch_model.pth`, `multi_branch_model.pth`, `merit_model.pth`, plus the merit normalizer file.

Plots go to `outputs/plots/<model>/`:

* `loss_curves.png`: train/validation loss and learning rate per epoch (only written when training)
* `confusion_matrix.png`: absolute and normalized counts
* `roc_curve.png`: ROC curve with the electron as positive class
* `probs_distribution.png`: per-class distribution of the classifier output
* `error_energy_distribution.png`: prediction error versus reconstructed energy, for protons and electrons
