# Imaging classifier for Fermi-LAT detector events.

This repository implements a PyTorch-based Convolutional Neural Network (CNN) to classify images of electron and proton events.
There is also a non-linear model to compare the CNN results to a more traditional method based on reconstructed variables (merit).

## Repository Architecture

The codebase is structured using object-oriented principles, across multiple files.

### Directory Structure

* **`main.py`**: The primary execution script. Orchestrates the initialization of models, data loaders and pipelines.
* **`src/`**: The core Python package containing the class-based code.
  * `config.py`: Contains the config class, defining training hyperparameters, batch sizes and dataset paths.
  * `data.py`: Implements memory-mapped HDF5 reads, binary search indexing for chunk lookups and multiprocessing safety for PyTorch workers.
  * `model.py`: Defines the CNN architecture used for used for extracting the best features from the event display images.
  * `training_loop.py`: Contains the training and validation steps logic.
  * `evaluator.py`: Manages the computation of metrics like accuracy and loss on the test sets.
  * `pipelines.py`: High-level orchestration classes tying together the data modules, models and training loops.
  * `logger.py` & `utils.py`: Centralized logging and helper functions.
* **`outputs/`**: Generated plots and models.
  * `models/`: (not tracked) Saved PyTorch model weights (`.pth` files).
  * `plots/`: Automatically generated performance visualizations, including ROC curves, confusion matrices and loss histories.

## Data Loading & Performance Optimizations

Because of the size of the datasets (more than 10 GB per 100k events), this classifier features pipelines that don't rely on RAM loading.
*   **Multiprocessing Safety:** Forces the `spawn` start method to prevent HDF5 C-library locks across parallel PyTorch workers.
*   **Lazy Loading:** Slices data directly from disk using `h5py` handles, avoiding RAM oversubscription.
<!--*    **Thread Management:** Designed to operate with uncompressed `.hdf5` chunks, offloading normalization mathematics to the GPU to prevent CPU thread thrashing during training.-->

## Usage

### 1. Environment Setup
A dedicated Conda environment is recommended to manage PyTorch and HDF5 dependencies.
```bash
conda create -n imaging_classifier python=3.10
conda activate imaging_classifier
conda install torch torchvision numpy h5py scikit-learn torchmetrics mlxtend
```

### 2. Training the Model
Execute the main script to initialize the `FermiDataModule`, load the CNN and begin the training loop.
```bash
python main.py --help
```
```bash
Fermi-LAT electron/proton classifier.

options:
  -h, --help       show this help message and exit
  --merit          Use the merit variables model.
  --multi-branch   Use the multi-branch CNN model.
  --single-branch  Use the single-branch CNN model.
  --train          Run the training loop on a newly instantiated model.
  -v, --verbose    Verbose option (set logger to debug mode).
```
The code always runs the evaluation on the given test dataset, using the model saved in `outputs/models/`. If you don't have one or you want to train a new one you need to use the option `--train`.

## Outputs & Metrics

Upon completing a run, the pipeline exports classification metrics to `outputs/plots/`:
* `roc_curve.png`: Receiver Operating Characteristic curve detailing electron/proton separation power.
* `confusion_matrix.png`: Absolute and normalized classification counts.
* `loss_curves.png`: Training vs. validation loss tracking across epochs.
