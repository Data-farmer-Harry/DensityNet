import os

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import torch

from model import densityNet


# load train/test/valid data
train_npz = np.load(os.path.join("data", "train.npz"))["data"]
test_npz = np.load(os.path.join("data", "test.npz"))["data"]
valid_path = os.path.join("data", "valid.npz")
valid_npz = np.load(valid_path)["data"] if os.path.exists(valid_path) else None

#Normalization
feat = train_npz[:, :-1]
label = train_npz[:, -1]
FEAT_MEAN = feat.mean(axis=0)
FEAT_STD = feat.std(axis=0)
FEAT_STD[FEAT_STD == 0] = 1.0
LABEL_MEAN = label.mean()
LABEL_STD = label.std()
if LABEL_STD == 0:
    LABEL_STD = 1.0

# load XGBoost model
bst = XGBRegressor()
bst.load_model(os.path.join("train/XGBoost", "xgb_model.json"))

# Load the neural network model
device = "cuda" if torch.cuda.is_available() else "cpu"
NN = densityNet(n_in=6, activation="ReLU").to(device)
state_dict = torch.load(os.path.join("train", "NN", "model.pth"), map_location=device)
NN.load_state_dict(state_dict)
NN.eval()


def pick_three(sorted_arr: np.ndarray) -> np.ndarray:

    if len(sorted_arr) <= 3:
        return sorted_arr
    return np.array([sorted_arr[0], sorted_arr[len(sorted_arr) // 2], sorted_arr[-1]])


def trim_zero_tail(r, y_true, y_pred, abs_err, keep=5, tol=0.0):
    zero_mask = np.isclose(y_true, 0.0, atol=tol, rtol=0.0)
    if not zero_mask.any():
        return r, y_true, y_pred, abs_err

    nonzero_idx = np.where(~zero_mask)[0]
    if nonzero_idx.size == 0:
        cutoff = min(len(y_true), keep)
        return r[:cutoff], y_true[:cutoff], y_pred[:cutoff], abs_err[:cutoff]

    first_nonzero = nonzero_idx[0]
    run_start = None
    run_len = 0
    cutoff = len(y_true)
    for idx in range(first_nonzero, len(zero_mask)):
        if zero_mask[idx]:
            if run_start is None:
                run_start = idx
                run_len = 1
            else:
                run_len += 1
            if run_len >= keep:
                cutoff = run_start + keep
                break
        else:
            run_start = None
            run_len = 0

    return r[:cutoff], y_true[:cutoff], y_pred[:cutoff], abs_err[:cutoff]


def main():
    sel_widths = np.array([4, 37, 56])
    sel_temps = np.array([270, 300, 328])

    # Merge all available sets (train/test/valid) for plotting.
    if valid_npz is not None:
        data_all = np.concatenate([train_npz, valid_npz, test_npz], axis=0)
    else:
        data_all = np.concatenate([train_npz, test_npz], axis=0)

    # Pre-calculate y_true, y_pred, and abs_err for 9 combinations to unify the coordinate range.
    cases = {}  # (i, j) -> dict
    all_y = []
    all_err = []

    for i, w in enumerate(sel_widths):
        for j, t in enumerate(sel_temps):
            mask = (np.isclose(data_all[:, 1], w)) & (np.isclose(data_all[:, 2], t))
            data = data_all[mask]
            if data.size == 0:
                cases[(i, j)] = None
                continue

            data = data[np.argsort(data[:, 0])]
            X_raw = data[:, :-1]
            y_true = data[:, -1]
            r = data[:, 0]

            y_pred = bst.predict(X_raw)
            abs_err = np.abs(y_pred - y_true)

            cases[(i, j)] = {
                "w": w,
                "t": t,
                "r": r,
                "y_true": y_true,
                "y_pred": y_pred,
                "abs_err": abs_err,
            }

            all_y.append(y_true)
            all_y.append(y_pred)
            all_err.append(abs_err)

    if not any(cases.values()):
        print("No valid (width, temperature) combinations found in test data.")
        return

    all_y_arr = np.concatenate(all_y) if all_y else np.array([0.0, 1.0])
    y_min = float(all_y_arr.min())
    y_max = float(all_y_arr.max())
    all_err_arr = np.concatenate(all_err) if all_err else np.array([0.0, 1.0])
    err_max = float(all_err_arr.max())

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(9, 10), dpi=300)

    gs = gridspec.GridSpec(
        8, 3, figure=fig, height_ratios=[5, 1, 0.8, 5, 1, 0.8, 5, 1], hspace=0.16
    )

    for i, w in enumerate(sel_widths):
        for j, t in enumerate(sel_temps):
            case = cases.get((i, j))
            row_top = 3 * j
            row_bottom = 3 * j + 1

            ax_top = fig.add_subplot(gs[row_top, i])
            ax_bottom = fig.add_subplot(gs[row_bottom, i], sharex=ax_top)

            if not case:
                ax_top.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                ax_top.set_xticks([])
                ax_top.set_yticks([])
                ax_bottom.set_xticks([])
                ax_bottom.set_yticks([])
                continue

            r = case["r"]
            y_true = case["y_true"]
            y_pred = case["y_pred"]
            abs_err = case["abs_err"]

            r, y_true, y_pred, abs_err = trim_zero_tail(
                r, y_true, y_pred, abs_err, keep=5, tol=0.0
            )

            # Upper part: Data points + Prediction curve
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
            ax_top.set_ylabel("Density", fontsize=8)
            if np.isclose(w, 4):
                ax_top.set_ylim(-0.2, y_max+0.5)
            else:
                ax_top.set_ylim(y_min, y_max+0.5)
            ax_top.legend(loc="upper right", fontsize=6, frameon=False)
            ax_top.set_title(f"w:{w} Å      T:{t}K", fontsize=8)

            # Bottom half: Absolute error (red), coordinates 0-1
            ax_bottom.plot(r, abs_err, color="red", linewidth=0.8, label="|Error|")
            ax_bottom.set_ylim(-0.2, 1)
            ax_bottom.legend(loc="upper right", fontsize=6, frameon=False)

            if j == len(sel_temps) - 1:
                ax_bottom.set_xlabel("Coord1", fontsize=8)
            else:
                ax_bottom.set_xlabel("")
            ax_bottom.set_ylabel("|Error|", fontsize=8)

            ax_top.tick_params(labelbottom=False)

            ax_top.tick_params(labelsize=6, axis="x", pad=1)
            ax_bottom.tick_params(labelsize=6, axis="x", pad=2)
            ax_bottom.tick_params(labelsize=6, axis="y", pad=1)

    plt.tight_layout()
    out_path = os.path.join("figure", "nine_3x3.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 3x3 panel figure with error to {out_path}")


if __name__ == "__main__":
    main()
