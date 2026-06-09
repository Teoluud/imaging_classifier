import torch
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path

from data import ImportData
from model import FermiMultiBranchCNN
from training_loop import TrainingLoop
from logger import logger


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    RANDOM_SEED = 42
    LEARNING_RATE = 0.001

    DATA_DIR = Path("/data1/grosjacques/data")
    proton_path = DATA_DIR / "dataset_chunk_allpro_no_norm.npz"
    electron_path = DATA_DIR / "dataset_chunk_allhee_no_norm.npz"

    data = ImportData(proton_path, electron_path)
    train_loader, val_loader = data.train_split(split=0.8, random_state=RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)
    model = FermiMultiBranchCNN()
    model.to(device)

    # summary(model=model, input_size=(1, 3, 113, 113))

    # --- TRAINING ---
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=0)
    acc_fn = MulticlassAccuracy(num_classes=2)

    training_loop = TrainingLoop(model, loss_fn, optimizer, acc_fn, device)

    epochs = 50

    training_loop.run(epochs, train_loader, val_loader)

    import matplotlib.pyplot as plt
    import numpy as np

    epoch_x = np.arange(0, epochs, 1)

    train_losses = np.array(training_loop.train_losses)
    val_losses = np.array(training_loop.val_losses)
    learning_rates = np.array(training_loop.learning_rates)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train and Validation Loss")
    plt.plot(epoch_x, train_losses, label="Train Loss")
    plt.plot(epoch_x, val_losses, label="Validation Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Learning Rate")
    plt.plot(epoch_x, learning_rates)

    plt.savefig("loss_curves.png")