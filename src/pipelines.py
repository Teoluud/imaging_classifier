from pathlib import Path

import torch
from torchmetrics.classification import MulticlassAccuracy

from src.config import Config
from src.logger import logger
from src.data import FermiDataModule
from src.training_loop import TrainingLoop
from src.utils import build_file_list, plot_training_results, plot_conf_matrix, plot_roc_curve
from src.evaluator import Evaluator


class ImagingPipeline:
    """ Orchestrates the data loading, training, and evaluation for CNN models.
    """

    def __init__(self, model: torch.nn.Module, config: Config, device: torch.device, train: bool, save_path: str | Path) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train = train
        self.save_path = Path(save_path)

    def run(self) -> None:
        logger.info("--- Initializing Imaging Pipeline ---")

        data_module = FermiDataModule(
            file_path=self.config.data_dir,
            batch_size=self.config.batch_size
        )

        loaders = data_module.train_split(
            train_split=self.config.train_split,
            test_split=self.config.test_split,
            random_state=self.config.random_seed
        )

        train_loader = loaders["train"]
        val_loader = loaders["val"]

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        acc_fn = MulticlassAccuracy(num_classes=2, average="micro")

        if self.train:
            trainer = TrainingLoop(
                model=self.model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=acc_fn,
                device=self.device,
                model_save_path=self.save_path
            )

            trainer.run(self.config.epochs, train_loader, val_loader)

            # Save evaluation metrics
            plot_training_results(
                epochs=self.config.epochs,
                train_losses=trainer.train_losses,
                val_losses=trainer.val_losses,
                learning_rates=trainer.learning_rates,
                save_path=self.config.plot_save_path,
                title=f"Training Results: {self.model._get_name()}"
            )

        # Evaluation phase
        # Load the saved model
        logger.info(f"Loading best weights from {self.save_path.name} for evaluation...")
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device, weights_only=True))

        evaluator = Evaluator(
            model=self.model,
            loss_fn=loss_fn,
            accuracy_fn=acc_fn,
            device=self.device
        )

        if self.config.test_split is not None:
            test_loader = loaders["test"]
            split_name = "Test"
        else:
            test_loader = val_loader
            split_name = "Validation"

        eval_metrics = evaluator.evaluate(data_loader=test_loader, split_name=split_name)

        preds = eval_metrics["preds"]
        truths = eval_metrics["truths"]
        probs = eval_metrics["probs"]

        # Calculate Recall
        correct_electrons = (preds == truths).sum().item()
        total_electrons = len(truths)
        if total_electrons > 0:
            electron_recall = (correct_electrons / total_electrons)
            logger.info(f"[{split_name}] Electron Recall: {electron_recall:.2%}")
        else:
            logger.warning(f"No electrons found in dataset to calculate recall.")

        plot_conf_matrix(preds, truths, self.config.class_names,
                         save_path=self.config.conf_matrix_save_path,
                         title=f"Confusion Matrix: {self.model._get_name()}, {split_name} dataset.")
        plot_roc_curve(probs, truths,
                       save_path=self.config.roc_curve_save_path,
                       title=f"ROC Curve: {self.model._get_name()}, {split_name} dataset.")
        logger.debug(f"Exported evaluation metrics to {self.config.conf_matrix_save_path} and {self.config.roc_curve_save_path}")


class MeritPipeline:
    """ Orchestrates the data loading, training, and evaluation for the merit variables models.
    """

    def __init__(self, model: torch.nn.Module, config: Config, device: torch.device, train:bool) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train = train
        

    def run(self) -> None:
        logger.info("--- Initializing Merit Variables Pipeline ---")
        
        merit_data_module = FermiDataModule(
            file_path=self.config.data_dir,
            batch_size=self.config.batch_size,
            merit=True
        )

        merit_loaders = merit_data_module.train_split(
            train_split=self.config.train_split, random_state=self.config.random_seed
        )
        merit_train_loader = merit_loaders["train"]
        merit_val_loader = merit_loaders["val"]

        loss_fn = torch.nn.CrossEntropyLoss()
        merit_optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config.merit_learning_rate,
            weight_decay=0
        )
        acc_fn = MulticlassAccuracy(num_classes=2, average="micro")

        if self.train:
            merit_trainer = TrainingLoop(
                model=self.model,
                loss_fn=loss_fn,
                optimizer=merit_optimizer,
                accuracy_fn=acc_fn,
                device=self.device,
                model_save_path=self.config.merit_model_save_path
            )

            merit_trainer.run(self.config.epochs, merit_train_loader, merit_val_loader)

            plot_training_results(
                epochs=self.config.epochs,
                train_losses=merit_trainer.train_losses,
                val_losses=merit_trainer.val_losses,
                learning_rates=merit_trainer.learning_rates,
                save_path=self.config.merit_plot_save_path,
                title=f"Training Results: {self.model._get_name()}"
            )

        self.model.load_state_dict(torch.load(self.config.merit_model_save_path, map_location=self.device, weights_only=True))
        merit_evaluator = Evaluator(
            model=self.model,
            loss_fn=loss_fn,
            accuracy_fn=acc_fn,
            device=self.device
        )

        merit_metrics = merit_evaluator.evaluate(data_loader=merit_val_loader, split_name="Merit Validation")
        
        test_loader = merit_data_module.get_test_dataset(
            file_path=self.config.test_data_dir,
            merit=True
        )

        split_name = "Test"
        merit_test_metrics = merit_evaluator.evaluate(data_loader=test_loader, split_name=split_name)

        merit_test_preds = merit_test_metrics["preds"]
        merit_test_truths = merit_test_metrics["truths"]
        merit_test_probs = merit_test_metrics["probs"]
        plot_conf_matrix(merit_test_preds, merit_test_truths, class_names=self.config.class_names,
                         save_path=self.config.merit_conf_matrix_save_path,
                         title=f"Confusion Matrix: {self.model._get_name()}, {split_name} dataset.")
        plot_roc_curve(merit_test_probs, merit_test_truths,
                       save_path=self.config.merit_roc_curve_save_path,
                       title=f"ROC Curve: {self.model._get_name()}, {split_name} dataset.")