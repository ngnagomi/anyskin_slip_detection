"""
Loads a model saved by train.py and evaluates it, 
printing the confusion matrix, precision, recall and F1.
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from data import build_dataset, load_from_hub, load_local_dir
from model import USDConvNetLite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="sample_data")
    p.add_argument("--from_hub", action="store_true")
    p.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.from_hub:
        df = load_from_hub()
    else:
        df = load_local_dir(args.data_dir)

    # Build windows with the same settings the model was trained with
    bundle = build_dataset(df, window=checkpoint["window"], stride=args.stride,
                           use_fft=checkpoint["use_fft"],
                           use_filter=checkpoint["use_filter"])

    # Normalize with the training data's mean and std
    X = (bundle.X - checkpoint["mu"]) / checkpoint["sigma"]

    model = USDConvNetLite(num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_t = torch.from_numpy(X).float().unsqueeze(1)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()

    print("Confusion matrix:")
    print(confusion_matrix(bundle.y, preds, labels=[0, 1]))
    print(classification_report(bundle.y, preds, labels=[0, 1],
                                target_names=["No-slip", "Slip"], zero_division=0))


if __name__ == "__main__":
    main()
