"""
Plots the leave-one-object-out confusion matrix for each configuration,
from results/confusion_<name>.npy.
"""

import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ["base", "fft", "filter", "fft_filter"]
LABELS = {
    "base": "USDConvNetLite",
    "fft": "USDConvNetLite + FFT",
    "filter": "USDConvNetLite + Filter",
    "fft_filter": "USDConvNetLite + FFT + Filter",
}
CLASSES = ["No-slip", "Slip"]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, name in zip(axes, CONFIGS):
        cm = np.load("results/confusion_" + name + ".npy")

        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASSES)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABELS[name])

        # Write the count in each cell (white text on dark cells)
        for i in range(2):
            for j in range(2):
                if cm[i, j] > cm.max() / 2:
                    color = "white"
                else:
                    color = "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=150)
    print("Saved results/confusion_matrices.png")


if __name__ == "__main__":
    main()
