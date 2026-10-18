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
main.py                  CLI entry point
environment.yaml         Conda environment
src/
  config.py              Hyperparameters, paths, one config class per model
  data/
    datasets.py          HDF5 datasets (images and merit variables), lazy reads
    data_module.py       Stratified split, DataLoaders, merit normalizer setup
    transforms.py        Transform protocols, image and merit normalization
  model.py               CNN and MLP architectures
  training_loop.py       Train/validation loop, best-model checkpointing
  evaluator.py           Evaluator and the EvalResults dataclass
  pipelines.py           ClassifierPipeline: data -> (train) -> evaluate -> plots
  plotting.py            Loss plot and the EvaluationPlotter class
  logger.py              tqdm-safe logger
outputs/                 Generated plots and weights
```

## Data

`data_dir` (in `config.py`) must contain one HDF5 file per chunk:

```
<data_dir>/protons/*.hdf5      -> label 0
<data_dir>/electrons/*.hdf5    -> label 1
```

Each file holds the datasets `view_x`, `view_y`, `view_top` (images), `merit_values` (17 variables per event) and `meta` (column 2 is the reconstructed energy in MeV).

Only the first `N_CLASS_EVENTS / CHUNK_SIZE` files (sorted by name) are used for each class. With the defaults in `datasets.py` (200000 events per class, 50000 events per file) that is 4 files per class. Change the two constants to use more or less data.

Datasets are never loaded into RAM. `FermiLATDataset` records the event range of every file, finds the file of a given index with a binary search, and reads the single event from an `h5py` handle that is opened lazily in each worker. Handles are dropped when the dataset is pickled, and `main.py` forces the `spawn` start method so HDF5 state is never shared between DataLoader workers. The DataLoaders use 8 workers (hardcoded in `data_module.py`).

`ImagingDataset` and `MeritDataset` implement `_read`, which returns the raw tensor for one event. The base class reads the `meta` row, applies the transform and attaches the label. `MeritDataset` also has `load_all()`, which returns every merit vector so that the normalizer can be fitted.

## Transforms

Every transform follows the `Transform` protocol in `src/data/transforms.py`:

```python
def __call__(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor: ...
```

`x` is the event data and `meta` is the event's `meta` row, so image and merit datasets share the same code path.

A transform whose parameters are computed from the training data satisfies the `FittedTransform` protocol: the same call signature, plus `fit(data)`, `save(path)` and `load(path)`. `NormalizerSettings(cls, save_path)` is a small frozen dataclass that says which fitted transform to use and where its state is stored.

- **`ImageLogNormalizer`** (stateless). Pixels with a positive value are scaled by 1000 (converting to keV), except for the tracker rows (the rows after the first 11 of the X and Y views), which use 5000 (mips to fC conversion). The top view is always scaled by 1000 since it doesn't contain TKR hits. Then `log10` is taken and divided by `log10` of the event's reconstructed energy in keV (floored at 1 keV). Empty pixels stay 0.
- **`MeritMinMaxNormalizer`** (fitted). Per-variable min-max scaling. The minimum and maximum are computed **on the training split only** and saved with `torch.save`. The denominator is clamped to `1e-8` for safety.

If a config has a `normalizer`, the pipeline calls `FermiDataModule.setup_merit_normalizer` after the split is created:

- with `--train` the normalizer is fitted on the training indices (`normalizer.cls.fit`) and saved to `normalizer.save_path`;
- without `--train` the saved normalizer is loaded, so evaluation always uses the statistics the weights were trained with.

It raises a `TypeError` if the dataset is not a `MeritDataset`, because fitting needs `load_all()`.

To add a transform, write any callable with the `(x, meta)` signature and set it as `transform` in a config. If it has fitted state, implement `fit`/`save`/`load` and set `normalizer=NormalizerSettings(YourClass, path)` instead.

## Setup

The code is run on Python 3.14.5. The environment is defined in `environment.yaml` (conda-forge) and installs `pytorch`, `h5py`, `matplotlib`, `tqdm`, `torchmetrics`, `mlxtend`, `torchinfo` and `argcomplete` (`numpy` comes in as a dependency).

```bash
conda env create -f environment.yaml
conda activate imaging_classifier
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

1. **Split.** Protons and electrons are shuffled separately (seed `random_seed`) and divided into train/validation/test (`train_split`, `val_split`, and the remainder), so class proportions are preserved. Defaults: 80% / 10% / 10%. If `val_split` is `None`, the remaining 20% is used as the validation set and evaluation runs on it.
2. **Training** (`--train`). Adam with `learning_rate` and `weight_decay`, cross-entropy loss, `epochs` epochs. The weights with the lowest validation loss are saved to `model_save_path`.
3. **Evaluation.** The saved weights are loaded and run over the test split. The `Evaluator` returns an `EvalResults` (loss, accuracy, probabilities, predictions, truths), the pipeline adds the event energies, and an `EvaluationPlotter` draws the plots below.

## Configuration

Edit `src/config.py`. A config class holds everything that is specific to a model, so the pipeline has no per-model branching:

- `dataset_cls`: `ImagingDataset` or `MeritDataset`.
- `transform`: the stateless transform applied by the dataset (`ImageLogNormalizer()` for the CNNs).
- `normalizer`: optional `NormalizerSettings` for a fitted transform (`MeritMinMaxNormalizer` for the merit model).
- `data_dir`, `model_save_path`, `plots_dir`: the data location and the per-model output paths.
- `hyper_params`: a `HyperParameters` dataclass. The defaults come from `default_hyper_params()`: `random_seed=42`, `learning_rate=1e-3`, `weight_decay=1e-4`, `batch_size=64`, `train_split=0.8`, `val_split=0.10`, `epochs=50`.

Every config instance gets its own `HyperParameters`, so changing one (for example with `--epochs`) never affects another. `MeritConfig` sets `weight_decay=0`.

## Outputs

Weights go to `outputs/models/` (not tracked): `single_branch_model.pth`, `multi_branch_model.pth`, `merit_model.pth`, plus the merit normalizer file (`merit_normalizer.pth`).

Plots go to `outputs/plots/<model>/`, where `<model>` is `singlebranch`, `multibranch` or `merit`:

- `loss_curves.png`: train/validation loss and learning rate per epoch (only written when training)
- `confusion_matrix.png`: absolute and normalized counts
- `roc_curve.png`: ROC curve with the electron as positive class
- `probs_distribution.png`: per-class distribution of the classifier output
- `error_energy_distribution.png`: prediction error versus reconstructed energy, for protons and electrons
