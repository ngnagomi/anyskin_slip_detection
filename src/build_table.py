"""
Builds a summary table of leave-one-object-out results for each configuration:
accuracy (mean +/- std over objects) and slip precision, recall and F1
(over all held-out windows combined).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train import slip_metrics

CONFIGS = ["base", "fft", "filter", "fft_filter"]
LABELS = {
    "base": "USDConvNetLite",
    "fft": "USDConvNetLite + FFT",
    "filter": "USDConvNetLite + Filter",
    "fft_filter": "USDConvNetLite + FFT + Filter",
}


def main():
    rows = []
    for name in CONFIGS:
        with open("results/loo_" + name + ".json") as f:
            results = json.load(f)

        accs = [r["accuracy"] for r in results]
        mean = np.mean(accs)
        std = np.std(accs)

        y_true, y_pred = [], []
        for r in results:
            y_true.extend(r["y_true"])
            y_pred.extend(r["y_pred"])
        m = slip_metrics(y_true, y_pred)

        rows.append({
            "config": LABELS[name],
            "loo_mean": mean,
            "loo_std": std,
            "n_objects": len(accs),
            "result": str(round(mean * 100, 2)) + " +/- " + str(round(std * 100, 2)) + "%",
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
        })

    table = pd.DataFrame(rows)
    print(table[["config", "result", "precision", "recall", "f1"]].to_string(index=False))
    table.to_csv("results/summary_table.csv", index=False)
    print("Saved results/summary_table.csv")


if __name__ == "__main__":
    main()
