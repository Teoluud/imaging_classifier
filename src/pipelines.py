from dataclasses import replace

import torch

from src.config import Config
from src.logger import logger
from src.data.data_module import FermiDataModule, Loader
from src.training_loop import TrainingLoop
from src.plotting import plot_training_results, EvaluationPlotter
from src.evaluator import Evaluator, EvalResults


class ClassifierPipeline:
    """Orchestrates data loading, training and evaluation for classifier models."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        device: torch.device,
        train: bool,
    ) -> None:
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._train = train

    def run(self) -> None:
        """Orchestrates the pipeline execution."""
        pipeline_name = self._config.__class__.__name__.removesuffix("Config")
        logger.info(f"--- Initializing {pipeline_name} Pipeline ---")

        loaders = self._prepare_data()

        if self._train:
            self._train_model(train_loader=loaders["train"], val_loader=loaders["val"])

        self._evaluate_model(loaders)

    def _prepare_data(self) -> dict[str, Loader]:
        """Initializes Dataset and returns DataLoaders."""
        dataset = self._config.dataset_cls(
            self._config.data_dir, transform=self._config.transform
        )
        self.data_module = FermiDataModule(
            dataset=dataset,
            batch_size=self._config.hyper_params.batch_size,
        )

        loaders = self.data_module.train_split(
            train_split=self._config.hyper_params.train_split,
            val_split=self._config.hyper_params.val_split,
            random_state=self._config.hyper_params.random_seed,
        )

        # Setup the merit normalizer if needed
        if self._config.normalizer is not None:
            self.data_module.setup_merit_normalizer(
                normalizer=self._config.normalizer,
                fit=self._train,
            )

        return loaders

    def _train_model(self, train_loader: Loader, val_loader: Loader) -> None:
        """Executes the training loop and plots losses."""
        optimizer = torch.optim.Adam(
            params=self._model.parameters(),
            lr=self._config.hyper_params.learning_rate,
            weight_decay=self._config.hyper_params.weight_decay,
        )

        trainer = TrainingLoop(
            model=self._model,
            loss_fn=self._config.loss_fn,
            optimizer=optimizer,
            accuracy_fn=self._config.accuracy_fn,
            device=self._device,
            model_save_path=self._config.model_save_path,
        )

        trainer.run(self._config.hyper_params.epochs, train_loader, val_loader)

        self._config.plots_dir.mkdir(parents=True, exist_ok=True)

        plot_training_results(
            epochs=self._config.hyper_params.epochs,
            train_losses=trainer.train_losses,
            val_losses=trainer.val_losses,
            learning_rates=trainer.learning_rates,
            save_path=self._config.loss_plot_save_path,
            title=f"Training Results: {self._model.__class__.__name__}",
        )

    def _evaluate_model(self, loaders: dict[str, Loader]) -> None:
        """Loads best weights, computes metrics and generates evaluation plots."""
        logger.info(f"Loading best weights from {self._config.model_save_path.name}...")
        self._model.load_state_dict(
            torch.load(
                self._config.model_save_path,
                map_location=self._device,
                weights_only=True,
            )
        )

        evaluator = Evaluator(
            model=self._model,
            loss_fn=self._config.loss_fn,
            accuracy_fn=self._config.accuracy_fn,
            device=self._device,
        )

        test_loader = (
            loaders["test"]
            if self._config.hyper_params.val_split is not None
            else loaders["val"]
        )
        split_name = (
            "Test" if self._config.hyper_params.val_split is not None else "Validation"
        )
        results = evaluator.evaluate(data_loader=test_loader, split_name=split_name)
        results = replace(
            results,
            energies=self.data_module.get_split_energies(
                "test" if self._config.hyper_params.val_split else "val"
            ),
        )
        self._log_and_plot_metrics(results, split_name)

    def _log_and_plot_metrics(self, results: EvalResults, split_name: str) -> None:
        """Handles plotting and saving of evaluation metrics."""
        preds, truths, probs = results.preds, results.truths, results.probs

        plotter = EvaluationPlotter(
            self._config.plots_dir,
            self._config.class_names,
            self._model.__class__.__name__,
            split_name,
        )

        plotter.plot_conf_matrix(preds, truths)
        plotter.plot_roc_curve(probs, truths)
        plotter.plot_probs_distribution(probs, truths)
        if results.energies is not None:
            plotter.plot_error_energy_distribution(probs, truths, results.energies)
            plotter.plot_energy_distribution(results.energies, truths)


        logger.debug(f"Exported evaluation metrics to{self._config.plots_dir}")
