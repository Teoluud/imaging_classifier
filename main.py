import torch
from torchmetrics.classification import MulticlassAccuracy
import argparse

from src.config import Config
from src.data import FermiDataModule
from src.model import FermiMultiBranchCNN, FermiMeritVarsNN
from src.training_loop import TrainingLoop
from src.utils import plot_training_results, plot_conf_matrix, plot_roc_curve
from src.evaluator import Evaluator
from src.logger import logger


def main(args) -> None:
    # Setup logger
    if args.verbose:
        logger.setLevel("DEBUG")

    # Initialize configurations
    config = Config()

    # Setup device-agnostic code
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.debug(f"Using device: {device}")

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
    merit_model = FermiMeritVarsNN().to(device)

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
    
    if args.merit:
        merit_optimizer = torch.optim.Adam(
            params=merit_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0
        )
        merit_data_module = FermiDataModule(
            proton_path=config.proton_path,
            electron_path=config.electron_path,
            batch_size=config.batch_size,
            merit=True
        )

        merit_train_loader, merit_val_loader = merit_data_module.train_split(
            split=config.train_split, random_state=config.random_seed
        )

        merit_trainer = TrainingLoop(
            model=merit_model,
            loss_fn=loss_fn,
            optimizer=merit_optimizer,
            accuracy_fn=acc_fn,
            device=device,
            model_save_path=config.merit_model_save_path
        )

        merit_trainer.run(config.epochs, merit_train_loader, merit_val_loader)

        plot_training_results(
            epochs=config.epochs,
            train_losses=merit_trainer.train_losses,
            val_losses=merit_trainer.val_losses,
            learning_rates=merit_trainer.learning_rates,
            save_path=config.merit_plot_save_path
        )

        merit_evaluator = Evaluator(
            model=merit_model,
            loss_fn=loss_fn,
            accuracy_fn=acc_fn,
            device=device
        )

        merit_metrics = merit_evaluator.evaluate(data_loader=merit_val_loader, split_name="Merit Validation")
        merit_preds = merit_metrics["preds"]
        merit_truths = merit_metrics["truths"]
        merit_probs = merit_metrics["probs"]
        plot_conf_matrix(merit_preds, merit_truths, class_names=config.class_names, save_path=config.merit_conf_matrix_save_path)
        plot_roc_curve(merit_probs, merit_truths, save_path=config.merit_roc_curve_save_path)
    # Evaluation phase
    # Load the saved model
    logger.info(f"Loading best weights from {config.model_save_path} for evaluation...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device, weights_only=True))

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=acc_fn,
        device=device
    )

    test_loader = data_module.get_test_dataset(
        proton_path=None,
        electron_path=config.test_electron_path
    )

    eval_metrics = evaluator.evaluate(data_loader=test_loader, split_name="Test")

    preds = eval_metrics["preds"]
    truths = eval_metrics["truths"]
    probs = eval_metrics["probs"]

    # Calculate Recall
    correct_electrons = (preds == truths).sum().item()
    total_electrons = len(truths)
    if total_electrons > 0:
        electron_recall = (correct_electrons / total_electrons)
        logger.info(f"Electron Recall: {electron_recall:.2%}")
    else:
        logger.warning(f"No electrons found in dataset to calculate recall.")

    plot_conf_matrix(preds, truths, config.class_names, save_path=config.conf_matrix_save_path)
    # plot_roc_curve(probs, truths, save_path=config.roc_curve_save_path)
    logger.debug(f"Exported evaluation metrics to {config.conf_matrix_save_path} and {config.roc_curve_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fermi-LAT electron/proton classifier.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose option (set logger to debug mode).")
    parser.add_argument("--train", action="store_true", help="Run the training loop on a newly instantiated model.")
    parser.add_argument("--merit", action="store_true", help="Run the whole pipeline for the Merit Variables network.")

    args = parser.parse_args()

    main(args)