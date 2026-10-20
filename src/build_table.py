"""
Builds a summary table of leave-one-object-out accuracy (mean +/- std)
for each configuration.
"""
import json

import numpy as np
import pandas as pd

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
        rows.append({
            "config": LABELS[name],
            "loo_mean": mean,
            "loo_std": std,
            "n_objects": len(accs),
            "result": str(round(mean * 100, 2)) + " +/- " + str(round(std * 100, 2)) + "%",
        })

    table = pd.DataFrame(rows)
    print(table[["config", "result", "n_objects"]].to_string(index=False))
    table.to_csv("results/summary_table.csv", index=False)
    print("Saved results/summary_table.csv")


if __name__ == "__main__":
    main()
