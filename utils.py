import numpy as np
import matplotlib.pyplot as plt


def plot_training_results(
        epochs: int,
        train_losses: list[float],
        val_losses: list[float],
        learning_rates: list[float],
        save_path: str = "loss_curves.png"
) -> None:
    """ Generates and saves optimization loss metrics.
    """
    epoch_x = np.arange(0, epochs, 1)

    plt.figure(figsize=(12, 6))

    # Loss curves subplot
    plt.subplot(1, 2, 1)
    plt.title("Train and Validation Loss")
    plt.plot(epoch_x, np.array(train_losses), label="Train Loss")
    plt.plot(epoch_x, np.array(val_losses), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Learning rate subplot
    plt.subplot(1, 2, 2)
    plt.title("Learning Rate")
    plt.plot(epoch_x, np.array(learning_rates))
    plt.xlabel("Epoch")
    plt.ylabel("LR")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 