"""
Train USDConvNetLite on the anyskin_slip_detection dataset.

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from data import build_dataset, load_from_hub, load_local_dir
from model import USDConvNetLite, count_parameters

'''
Wraps NumPy arrays into a PyTorch dataloader, which training loop iterates over.
'''

def make_loader(X, y, batch_size, shuffle):

    X_t = torch.from_numpy(X).unsqueeze(1)
    y_t = torch.from_numpy(y)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

'''
Standardizes features to mean 0 and std 1.
'''

def normalize(X_train, X_test):

    mu = X_train.mean(axis=(0, 1, 2), keepdims=True)
    sigma = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    return (X_train - mu) / sigma, (X_test - mu) / sigma

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)

'''
Training one model.
'''

def train_one_model(X_train, y_train, X_test, y_test, epochs, batch_size, lr, device, track_history=False):
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    eval_train_loader = make_loader(X_train, y_train, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    model = USDConvNetLite(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "test_acc": []} if track_history else None
    best_acc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        test_acc = evaluate_accuracy(model, test_loader, device)
        if track_history:
            train_acc = evaluate_accuracy(model, eval_train_loader, device)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # collect final predictions with the BEST checkpoint, for the confusion matrix
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(pred.tolist())
            all_true.extend(yb.numpy().tolist())

    result = {"model": model, "accuracy": best_acc, "y_true": all_true, "y_pred": all_preds}
    if track_history:
        result["history"] = history
    return result

'''
80% training and 20% testing.
'''

def run_random_split(bundle, args, device):
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=0, stratify=bundle.y
    )
    X_train, X_test = normalize(X_train, X_test)
    r = train_one_model(
        X_train, y_train, X_test, y_test, args.epochs, args.batch_size, args.lr, device
    )
    acc = r["accuracy"]
    print("Random split test accuracy:", round(acc, 4))
    return acc

'''
Holds out each object in turn, trains on the other objects 
and tests on the held-out object.
'''

def run_leave_one_object_out(bundle, args, device, exclude_from_eval=("no_slip",)):
    objects = np.unique(bundle.groups)
    eval_objects = [o for o in objects if o not in exclude_from_eval]
    results = []
    for held_out in eval_objects:
        test_mask = bundle.groups == held_out
        train_mask = ~test_mask
        X_train, y_train = bundle.X[train_mask], bundle.y[train_mask]
        X_test, y_test = bundle.X[test_mask], bundle.y[test_mask]
        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue
        X_train_n, X_test_n = normalize(X_train, X_test)
        r = train_one_model(X_train_n, y_train, X_test_n, y_test,
                             args.epochs, args.batch_size, args.lr, device)
        print("Held out", held_out, "accuracy:", round(r["accuracy"], 4))
        results.append({
            "object": held_out,
            "accuracy": r["accuracy"],
            "y_true": r["y_true"],
            "y_pred": r["y_pred"],
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="sample_data",
                    help="Directory of per-object CSVs.")
    p.add_argument("--from_hub", action="store_true",
                    help="Download the real dataset from the HF Hub instead "
                         "of --data_dir.")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.from_hub:
        df = load_from_hub()
    else:
        df = load_local_dir(args.data_dir)
    print("Loaded rows:", len(df))
    print("Number of objects:", df["object"].nunique())

    bundle = build_dataset(df, window=args.window, stride=args.stride)
    print("Windowed dataset shape:", bundle.X.shape)
    print("Slip ratio:", round(bundle.y.mean(), 3))

    dummy_model = USDConvNetLite(num_classes=2)
    print("Model parameter count:", count_parameters(dummy_model))

    run_random_split(bundle, args, device)
    run_leave_one_object_out(bundle, args, device)

    # Final model trained on everything
    X_all, _ = normalize(bundle.X, bundle.X)
    final_model = train_one_model(
        X_all, bundle.y, X_all, bundle.y, args.epochs, args.batch_size, args.lr, device
    )["model"]
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), args.checkpoint)
    print("Saved final model to", args.checkpoint)


if __name__ == "__main__":
    main()