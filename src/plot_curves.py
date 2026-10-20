"""
Plots train vs test accuracy per epoch for each configuration,
from results/history_<name>.json.
"""
import json

import matplotlib.pyplot as plt

CONFIGS = ["base", "fft", "filter", "fft_filter"]
LABELS = {
    "base": "USDConvNetLite",
    "fft": "USDConvNetLite + FFT",
    "filter": "USDConvNetLite + Filter",
    "fft_filter": "USDConvNetLite + FFT + Filter",
}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, name in zip(axes.flat, CONFIGS):
        with open("results/history_" + name + ".json") as f:
            history = json.load(f)

        epochs = range(1, len(history["train_acc"]) + 1)
        ax.plot(epochs, history["train_acc"], label="Train")
        ax.plot(epochs, history["test_acc"], label="Test")
        ax.set_title(LABELS[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.2, 1.0)
        ax.legend()

    plt.tight_layout()
    plt.savefig("results/accuracy_curves.png", dpi=150)
    print("Saved results/accuracy_curves.png")


if __name__ == "__main__":
    main()
