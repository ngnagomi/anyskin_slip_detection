"""
Train USDConvNetLite on the anyskin_slip_detection dataset.

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from data import build_dataset, load_from_hub, load_local_dir
from model import USDConvNetLite, count_parameters

torch.set_num_threads(1)

'''
Fixes the random seeds so results can be reproduced.
'''

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

'''
Precision, recall and F1 for the slip class.
'''

def slip_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

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
Returns the model's predicted class for every window in a loader.
'''

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb.to(device)).argmax(dim=1).cpu().tolist())
    return preds

'''
Training one model.
If a validation set is given, keeps the epoch with the best slip F1 on it.
Otherwise keeps the last epoch. The test set is never used to choose the epoch.
'''

def train_one_model(X_train, y_train, X_test, y_test, epochs, batch_size, lr, device,
                    track_history=False, X_val=None, y_val=None):
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    eval_train_loader = make_loader(X_train, y_train, batch_size, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size, shuffle=False)
    if X_val is not None:
        val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    model = USDConvNetLite(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "test_acc": []} if track_history else None
    best_f1, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        if track_history:
            history["train_acc"].append(evaluate_accuracy(model, eval_train_loader, device))
            history["test_acc"].append(evaluate_accuracy(model, test_loader, device))

        if X_val is not None:
            val_f1 = slip_metrics(y_val, predict(model, val_loader, device))["f1"]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predictions on the test set with the chosen model, for the metrics and confusion matrix
    all_preds = predict(model, test_loader, device)
    all_true = y_test.tolist()
    accuracy = float(np.mean(np.array(all_preds) == y_test))

    result = {"model": model, "accuracy": accuracy, "y_true": all_true, "y_pred": all_preds}
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
In each fold, N_VAL_OBJECTS of the training objects are set aside
as validation, used to choose the best epoch.
'''

N_VAL_OBJECTS = 2

def run_leave_one_object_out(bundle, args, device, exclude_from_eval=("no_slip",)):
    objects = np.unique(bundle.groups)
    eval_objects = [o for o in objects if o not in exclude_from_eval]
    results = []
    for held_out in eval_objects:
        # Pick validation objects from the remaining objects with slips
        candidates = [o for o in eval_objects if o != held_out]
        val_objects = np.random.choice(candidates, size=N_VAL_OBJECTS, replace=False)

        test_mask = bundle.groups == held_out
        val_mask = np.isin(bundle.groups, val_objects)
        train_mask = ~test_mask & ~val_mask
        X_train, y_train = bundle.X[train_mask], bundle.y[train_mask]
        X_val, y_val = bundle.X[val_mask], bundle.y[val_mask]
        X_test, y_test = bundle.X[test_mask], bundle.y[test_mask]
        if len(np.unique(y_train)) < 2 or len(X_test) == 0:
            continue

        # Normalize with training statistics only
        X_train_n, X_val_n = normalize(X_train, X_val)
        _, X_test_n = normalize(X_train, X_test)
        r = train_one_model(X_train_n, y_train, X_test_n, y_test,
                             args.epochs, args.batch_size, args.lr, device,
                             X_val=X_val_n, y_val=y_val)
        m = slip_metrics(r["y_true"], r["y_pred"])
        print("Held out", held_out,
              "| accuracy:", round(r["accuracy"], 4),
              "| precision:", round(m["precision"], 4),
              "| recall:", round(m["recall"], 4),
              "| F1:", round(m["f1"], 4))
        results.append({
            "object": held_out,
            "val_objects": val_objects.tolist(),
            "accuracy": r["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
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
    mu = float(bundle.X.mean())
    sigma = float(bundle.X.std()) + 1e-6
    X_all = (bundle.X - mu) / sigma
    final_model = train_one_model(
        X_all, bundle.y, X_all, bundle.y, args.epochs, args.batch_size, args.lr, device
    )["model"]

    # Save the weights together with the normalization and window settings,
    # so evaluate.py can process new data exactly the same way
    checkpoint = {
        "state_dict": final_model.state_dict(),
        "mu": mu,
        "sigma": sigma,
        "window": args.window,
        "use_fft": True,
        "use_filter": True,
    }
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    print("Saved final model to", args.checkpoint)


if __name__ == "__main__":
    main()