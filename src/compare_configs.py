"""
Compare the four processing configurations of USDConvNetLite:
base, FFT, high-pass filter, and FFT + filter.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from data import build_dataset, load_from_hub, load_local_dir
from train import normalize, run_leave_one_object_out, set_seed, train_one_model

CONFIGS = [
    ("base", False, False),
    ("fft", True, False),
    ("filter", False, True),
    ("fft_filter", True, True),
]

'''
Saves data to a JSON file.
'''

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

'''
Trains on a random 80/20 split and saves accuracy per epoch,
used for the accuracy curves.
'''

def run_history(bundle, args, device, name):
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=0, stratify=bundle.y
    )
    X_train, X_test = normalize(X_train, X_test)
    r = train_one_model(X_train, y_train, X_test, y_test,
                        args.epochs, args.batch_size, args.lr, device,
                        track_history=True)
    save_json(r["history"], "results/history_" + name + ".json")

'''
Runs leave-one-object-out and saves per-object results
and one confusion matrix combining all held-out objects.
'''

def run_loo(bundle, args, device, name):
    results = run_leave_one_object_out(bundle, args, device)
    save_json(results, "results/loo_" + name + ".json")

    y_true, y_pred = [], []
    for r in results:
        y_true.extend(r["y_true"])
        y_pred.extend(r["y_pred"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.save("results/confusion_" + name + ".npy", cm)
    print("Confusion matrix:")
    print(cm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="sample_data")
    p.add_argument("--from_hub", action="store_true")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.from_hub:
        df = load_from_hub()
    else:
        df = load_local_dir(args.data_dir)
    Path("results").mkdir(exist_ok=True)

    for name, use_fft, use_filter in CONFIGS:
        print()
        print("Config:", name, "| FFT:", use_fft, "| Filter:", use_filter)
        # Same seed for every config, so they are compared fairly
        set_seed(args.seed)
        bundle = build_dataset(df, window=args.window, stride=args.stride,
                               use_fft=use_fft, use_filter=use_filter)
        run_history(bundle, args, device, name)
        run_loo(bundle, args, device, name)


if __name__ == "__main__":
    main()
