from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import Config, MeritConfig
from src.logger import logger
from src.data import FermiDataModule
from src.training_loop import TrainingLoop
from src.utils import plot_training_results, plot_conf_matrix, plot_roc_curve, plot_probs_distribution, plot_error_energy_distribution
from src.evaluator import Evaluator


class ClassifierPipeline:
    """ Orchestrates data loading, training and evaluation for classifier models.
    """

    def __init__(self, model: torch.nn.Module, config: Config, device: torch.device, train: bool, merit: bool) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.train = train
        self.merit = merit

    def run(self) -> None:
        """ Orchestrates the pipeline execution.
        """
        pipeline_type = "Merit Variables" if self.merit else "Imaging"
        logger.info(f"--- Initializing {pipeline_type} Pipeline ---")

        loaders = self._prepare_data()

        if self.train:
            self._train_model(train_loader=loaders["train"], val_loader=loaders["val"])

        self._evaluate_model(loaders)


    def _prepare_data(self) -> dict[str, DataLoader[Any]]:
        """ Initializes Dataset and returns DataLoaders. """
        self.data_module = FermiDataModule(
            file_path=self.config.data_dir,
            batch_size=self.config.batch_size,
            merit=self.merit
        )

        loaders = self.data_module.train_split(
            train_split=self.config.train_split,
            test_split=self.config.test_split,
            random_state=self.config.random_seed
        )

        # Setup the merit normalizer if needed
        if self.merit:
            # Check the config is MeritConfig (has normalizer save path)
            assert isinstance(self.config, MeritConfig)
            self.data_module.setup_merit_normalizer(
                save_path=self.config.normalizer_save_path,
                fit = self.train
            )

        return loaders

    def _train_model(self, train_loader: DataLoader[Any], val_loader: DataLoader[Any]) -> None:
        """ Executes the training loop and plots losses.
        """
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        trainer = TrainingLoop(
            model=self.model,
            loss_fn=self.config.loss_fn,
            optimizer=optimizer,
            accuracy_fn=self.config.accuracy_fn,
            device=self.device,
            model_save_path=self.config.model_save_path
        )

        trainer.run(self.config.epochs, train_loader, val_loader)

        plot_training_results(
            epochs=self.config.epochs,
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            learning_rates=trainer.learning_rates,
            save_path=self.config.loss_plot_save_path,
            title=f"Training Results: {self.model.__class__.__name__}"
        )

    def _evaluate_model(self, loaders: dict[str, DataLoader[Any]]) -> None:
        """ Loads best weights, computes metrics and generates evaluation plots.
        """
        logger.info(f"Loading best weights from {self.config.model_save_path.name}...")
        self.model.load_state_dict(torch.load(self.config.model_save_path, map_location=self.device, weights_only=True))

        evaluator = Evaluator(
            model=self.model,
            loss_fn=self.config.loss_fn,
            accuracy_fn=self.config.accuracy_fn,
            device=self.device
        )

        test_loader = loaders["test"] if self.config.test_split is not None else loaders["val"]
        split_name = "Test" if self.config.test_split is not None else "Validation"
        metrics = evaluator.evaluate(data_loader=test_loader, split_name=split_name)
        metrics["energies"] = self.data_module.get_split_energies("test" if self.config.test_split else "val")
        self._log_and_plot_metrics(metrics, split_name)

    def _log_and_plot_metrics(self, metrics: dict[str, Any], split_name: str) -> None:
        """ Handles classification metrics calculations and plotting.
        """
        preds, truths, probs = metrics["preds"], metrics["truths"], metrics["probs"]

        plot_conf_matrix(preds, truths, self.config.class_names,
                         save_path=self.config.conf_matrix_save_path,
                         title=f"Confusion Matrix: {self.model.__class__.__name__}, {split_name} dataset.")
        
        plot_roc_curve(probs, truths,
                       save_path=self.config.roc_curve_save_path,
                       title=f"ROC Curve: {self.model.__class__.__name__}, {split_name} dataset.")

        plot_probs_distribution(probs, truths,
                                save_path=self.config.probs_distribution_save_path,
                                title=f"Probs distribution: {self.model.__class__.__name__}, {split_name} dataset.")

        plot_error_energy_distribution(probs, truths, metrics["energies"],
                                       save_path=self.config.error_energy_distribution_save_path,
                                       title=f"Error vs Energy: {self.model.__class__.__name__}, {split_name} dataset.")

        # plot_energy_distribution(metrics["energies"], truths)

        logger.debug(f"Exported evaluation metrics to{self.config.conf_matrix_save_path} and {self.config.roc_curve_save_path}")