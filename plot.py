import os

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import torch

from model import densityNet


# Load train/test data
train_npz = np.load(os.path.join("data", "train.npz"))["data"]
test_npz = np.load(os.path.join("data", "test.npz"))["data"]

# Normalization parameters consistent with train.py
feat = train_npz[:, :-1]
label = train_npz[:, -1]
FEAT_MEAN = feat.mean(axis=0)
FEAT_STD = feat.std(axis=0)
FEAT_STD[FEAT_STD == 0] = 1.0
LABEL_MEAN = label.mean()
LABEL_STD = label.std()
if LABEL_STD == 0:
    LABEL_STD = 1.0

# XGBoost 
bst = XGBRegressor()
bst.load_model(os.path.join("train/XGboost", "xgb_model.json"))

# nn
device = "cuda" if torch.cuda.is_available() else "cpu"
NN = densityNet(n_in=6, activation="ReLU").to(device)
state_dict = torch.load(os.path.join("train", "NN", "model.pth"), map_location=device)
NN.load_state_dict(state_dict)
NN.eval()


def plot_single(width: float, temperature: float, save_dir: str = "figure") -> None:
    """
    Using the test set data as true values, the following plots are drawn:

    - Data (test set): solid blue line
    - XGBoost prediction: solid orange line
    - NN prediction: solid green line

    The horizontal axis is Coord1 (second column from source file),
    and the vertical axis is density (last column).
    """
    w = float(width)
    t = float(temperature)

    mask = (np.isclose(test_npz[:, 1], w)) & (np.isclose(test_npz[:, 2], t))
    data = test_npz[mask]
    if data.size == 0:
        print(f"No test data found for width={w}, temperature={t}")
        return


    data = data[np.argsort(data[:, 0])]

    X_raw = data[:, :-1]
    y_md = data[:, -1]
    r = data[:, 0]

    # XGBoost 
    y_xgb = bst.predict(X_raw)

    # NN 
    X_norm = (X_raw - FEAT_MEAN) / FEAT_STD
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_nn_norm = NN(X_tensor).cpu().numpy().reshape(-1)
    y_nn = y_nn_norm * LABEL_STD + LABEL_MEAN

    os.makedirs(save_dir, exist_ok=True)
    fname = f"width_{w}_temp_{t}.png"
    save_path = os.path.join(save_dir, fname)

    plt.figure(figsize=(4.0, 3.0), dpi=600)
    plt.plot(r, y_md, marker="o", linewidth=0.8, markersize=2.0, label="Data")
    plt.plot(r, y_xgb, marker="s", linewidth=0.8, markersize=2.0, label="XGBoost")
    plt.plot(r, y_nn, marker="x", linewidth=0.8, markersize=2.0, label="NN")
    plt.xlabel("Coord1")
    plt.ylabel("Density")
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    #Plot the (width, temperature) combinations for all test sets.
    os.makedirs("figure", exist_ok=True)

    pairs = np.unique(test_npz[:, [1, 2]], axis=0)
    for w, t in pairs:
        plot_single(w, t, save_dir="figure/test_figure")

