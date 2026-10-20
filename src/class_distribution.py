"""
Counts slip and no-slip rows for each object in the dataset,
to show the class imbalance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import load_from_hub


def main():
    df = load_from_hub()

    table = df.groupby("object")["slip"].value_counts().unstack(fill_value=0)
    table = table.rename(columns={0: "No-slip", 1: "Slip"})
    table = table[["Slip", "No-slip"]]
    table.columns.name = None
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    table["Slip %"] = (table["Slip"] / table["Total"] * 100).round(1)

    print(table)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/class_distribution.csv")
    print("Saved results/class_distribution.csv")


if __name__ == "__main__":
    main()
