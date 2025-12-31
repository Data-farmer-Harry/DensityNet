import os

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import torch

from model import densityNet


# load train / test data
train_npz = np.load(os.path.join("data", "train.npz"))["data"]
test_npz = np.load(os.path.join("data", "test.npz"))["data"]

# Normalization parameters consistent with train.py (for NN, if needed)
feat = train_npz[:, :-1]
label = train_npz[:, -1]
FEAT_MEAN = feat.mean(axis=0)
FEAT_STD = feat.std(axis=0)
FEAT_STD[FEAT_STD == 0] = 1.0
LABEL_MEAN = label.mean()
LABEL_STD = label.std()
if LABEL_STD == 0:
    LABEL_STD = 1.0

# Loading XGBoost models
bst = XGBRegressor()
bst.load_model(os.path.join("train/XGBoost", "xgb_model.json"))

# nn model for future expansion
device = "cuda" if torch.cuda.is_available() else "cpu"
NN = densityNet(n_in=6, activation="ReLU").to(device)
state_dict = torch.load(os.path.join("train", "NN", "model.pth"), map_location=device)
NN.load_state_dict(state_dict)
NN.eval()


def plot_case(width: float, temperature: float, save_dir: str = os.path.join("figure", "error")) -> None:

    w = float(width)
    t = float(temperature)

    mask = (np.isclose(test_npz[:, 1], w)) & (np.isclose(test_npz[:, 2], t))
    data = test_npz[mask]
    if data.size == 0:
        print(f"No test data found for width={w}, temperature={t}")
        return

    # Sort by Coord1
    data = data[np.argsort(data[:, 0])]

    X_raw = data[:, :-1]
    y_true = data[:, -1]
    r = data[:, 0]

    # XGBoost predict
    y_pred = bst.predict(X_raw)
    abs_err = np.abs(y_pred - y_true) 

    os.makedirs(save_dir, exist_ok=True)
    fname = f"width_{w}_temp_{t}.png"
    save_path = os.path.join(save_dir, fname)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(4.0, 3.0),
        dpi=600,
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08},
    )

    # Upper part: Predicted curve + Actual points
    ax_top.plot(r, y_pred, color="red", linewidth=1.0, label="XGBoost")
    ax_top.scatter(
        r,
        y_true,
        facecolors="none",
        edgecolors="blue",
        s=8,
        linewidths=0.6,
        label="Data",
    )
    ax_top.set_ylabel("Density")
    ax_top.legend(frameon=False, fontsize=8)
    ax_top.set_title(f"w:{w}Å      T:{t}K", fontsize=8)

    # absolute error
    ax_bottom.plot(r, abs_err, color="black", linewidth=0.8)
    ax_bottom.set_xlabel("Coord1")
    ax_bottom.set_ylabel("|Error|")


    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    # plot
    base_dir = os.path.join("figure", "error")
    os.makedirs(base_dir, exist_ok=True)

    pairs = np.unique(test_npz[:, [1, 2]], axis=0)
    for w, t in pairs:
        plot_case(w, t, save_dir=base_dir)
