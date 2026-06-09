import torch
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy
from pathlib import Path
from tqdm.auto import tqdm

from data import ImportData
from model import FermiMultiBranchCNN
from training import TrainingLoop
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
                                 weight_decay=1e-4)
    acc_fn = MulticlassAccuracy(num_classes=2)

    training_loop = TrainingLoop(model, loss_fn, acc_fn, device)

    epochs = 3

    for epoch in tqdm(range(epochs)):
        tqdm.write(f"\nEpoch: {epoch}\n------")
        train_loss = training_loop.train_step(train_loader, optimizer)

        val_loss = training_loop.validation_step(val_loader)

    logger.info("\nTraining Complete!")