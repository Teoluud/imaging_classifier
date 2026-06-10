import torch
from torchmetrics.classification import MulticlassAccuracy
import argparse

from config import Config
from data import FermiDataModule
from model import FermiMultiBranchCNN
from training_loop import TrainingLoop
from utils import plot_training_results


def main(args) -> None:
    # Initialize configurations
    config = Config()

    # Setup device-agnostic code
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    data_module = FermiDataModule(
        proton_path=config.proton_path,
        electron_path=config.electron_path,
        batch_size=config.batch_size
    )

    train_loader, val_loader = data_module.train_split(
        split=config.train_split,
        random_state=config.random_seed
    )

    # Define model
    model = FermiMultiBranchCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    acc_fn = MulticlassAccuracy(num_classes=2)

    if args.train:
        trainer = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=acc_fn,
            device=device,
            model_save_path=config.model_save_path
        )

        trainer.run(config.epochs, train_loader, val_loader)

        # Save evaluation metrics
        plot_training_results(
            epochs=config.epochs,
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            learning_rates=trainer.learning_rates,
            save_path=config.plot_save_path
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fermi-LAT electron/proton classifier.")
    parser.add_argument("--train", action="store_true", help="Run the training loop on a newly instantiated model.")

    args = parser.parse_args()

    main(args)