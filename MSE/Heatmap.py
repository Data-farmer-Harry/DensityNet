import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, "mse_matrix_xgb(new).xlsx")

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"{excel_path} not found")

    # # Rows represent width, columns represent temperature
    df = pd.read_excel(excel_path, index_col=0)

    # Missing values ​​are filled with 0.
    df_filled = df.fillna(0.0)
    data = df_filled.to_numpy(dtype=float)

    widths = df_filled.index.to_numpy()
    temps = df_filled.columns.to_numpy()
    n_rows, n_cols = data.shape

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    positive_vals = data[data > 0]
    if positive_vals.size > 0:
        vmax = float(np.percentile(positive_vals, 99))
    else:
        vmax = 1.0
    if vmax <= 0.0:
        vmax = 1.0

    data_plot = np.clip(data, 0.0, vmax)
    norm = PowerNorm(gamma=1.6, vmin=0.0, vmax=vmax)
    im = ax.imshow(data_plot, cmap="coolwarm", origin="lower", aspect="auto", norm=norm)

    temps_val = temps.astype(float)
    widths_val = widths.astype(float)

    max_ticks = 5

    # x
    n_temp_ticks = min(n_cols, max_ticks)
    temp_idx = np.linspace(0, n_cols - 1, n_temp_ticks, dtype=int)
    temp_idx = np.unique(temp_idx)
    ax.set_xticks(temp_idx)
    ax.set_xticklabels([f"{temps_val[i]:.0f}" for i in temp_idx], rotation=45, ha="right")

    # y
    n_width_ticks = min(n_rows, max_ticks)
    width_idx = np.linspace(0, n_rows - 1, n_width_ticks, dtype=int)
    width_idx = np.unique(width_idx)
    ax.set_yticks(width_idx)
    ax.set_yticklabels([f"{widths_val[i]:.1f}" for i in width_idx])

    ax.set_xlabel("Temperature(K)")
    ax.set_ylabel("Channel Width(Å)")

    # ===== Validation/Test Set Regions Based on the Logical Marking in sample.py =====
    # Temperature Division:
    # - Validation Set Temperature: 10% on both sides (approximately 20% total)
    # - Test Set Temperature: 45% of the remaining temperatures

    n_temp = n_cols
    if n_temp >= 5:
        k_val = max(1, int(round(n_temp * 0.1)))
        if 2 * k_val >= n_temp:
            k_val = 1
        val_temp_idx = list(range(k_val)) + list(range(n_temp - k_val, n_temp))
    else:
        val_temp_idx = [0]
        if n_temp > 1:
            val_temp_idx.append(n_temp - 1)

    val_temp_mask = np.zeros(n_temp, dtype=bool)
    val_temp_mask[val_temp_idx] = True

    mid_temp_idx = [i for i in range(n_temp) if not val_temp_mask[i]]
    m = len(mid_temp_idx)
    test_temp_idx = []
    if m > 0:
        k_test = max(1, int(round(m * 0.45)))  # 中间 45%
        if k_test >= m:
            test_temp_idx = mid_temp_idx
        else:
            start = (m - k_test) // 2
            test_temp_idx = mid_temp_idx[start : start + k_test]

    test_temp_mask = np.zeros(n_temp, dtype=bool)
    test_temp_mask[test_temp_idx] = True

    n_width = n_rows
    k_w = max(1, int(round(n_width * 0.4)))
    if k_w >= n_width:
        mid_width_idx = list(range(n_width))
    else:
        start_w = (n_width - k_w) // 2
        mid_width_idx = list(range(start_w, start_w + k_w))

    mid_width_mask = np.zeros(n_width, dtype=bool)
    mid_width_mask[mid_width_idx] = True

    if n_temp >= 2 and k_val >= 1 and (n_temp - k_val) >= 1:
        b1 = k_val - 0.5
        b2 = (n_temp - k_val) - 0.5

        ax.axvline(b1, color="k", linestyle="--", linewidth=0.7, zorder=4)
        ax.axvline(b2, color="k", linestyle="--", linewidth=0.7, zorder=4)


        trans = ax.get_xaxis_transform()
        y_text = 1.02  # Move it up a little (adjustable from 1.01 to 1.08)

        left_center = (0 + (k_val - 1)) / 2
        mid_start = k_val
        mid_end = n_temp - k_val - 1
        mid_center = (mid_start + mid_end) / 2 if mid_end >= mid_start else (n_temp - 1) / 2
        right_start = n_temp - k_val
        right_center = (right_start + (n_temp - 1)) / 2

        ax.text(left_center, y_text, "Extrapolate", transform=trans,
                ha="center", va="bottom", fontsize=8, clip_on=False)
        ax.text(mid_center, y_text, "Interpolate", transform=trans,
                ha="center", va="bottom", fontsize=8, clip_on=False)
        ax.text(right_center, y_text, "Extrapolate", transform=trans,
                ha="center", va="bottom", fontsize=8, clip_on=False)
    else:
        ax.set_title("Extrapolate / Interpolate / Extrapolate", fontsize=8)

    # Validation set: Entire column of light-colored
    for c in range(n_temp):
        if val_temp_mask[c]:
            rect = Rectangle(
                (c - 0.5, -0.5),
                1.0,
                n_rows,
                facecolor="white",
                alpha=0.45,
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

    # Dashed border for validation set: Draw a border around the two validation column groups
    if val_temp_idx:
        val_temp_idx_sorted = sorted(val_temp_idx)
        seg_start = val_temp_idx_sorted[0]
        prev = seg_start
        segments = []
        for idx in val_temp_idx_sorted[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            segments.append((seg_start, prev))
            seg_start = idx
            prev = idx
        segments.append((seg_start, prev))

        for c_start, c_end in segments:
            border_val = Rectangle(
                (c_start - 0.5, -0.5),
                c_end - c_start + 1,
                n_rows,
                fill=False,
                edgecolor="k",
                linestyle="--",
                linewidth=0.7,
                zorder=3,
            )
            ax.add_patch(border_val)

    if mid_width_idx and test_temp_idx:
        row_start = min(mid_width_idx) - 0.5
        row_height = max(mid_width_idx) - min(mid_width_idx) + 1
        for c in test_temp_idx:
            rect = Rectangle(
                (c - 0.5, row_start),
                1.0,
                row_height,
                facecolor="white",
                alpha=0.25,
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

        # Draw a dashed boundary around the test set only
        col_start = min(test_temp_idx)
        col_end = max(test_temp_idx)
        border = Rectangle(
            (col_start - 0.5, row_start),
            col_end - col_start + 1,
            row_height,
            fill=False,
            edgecolor="k",
            linestyle="--",
            linewidth=0.7,
            zorder=3,
        )
        ax.add_patch(border)

    cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(0, vmax, 6))
    cbar.set_label("MSE")

    plt.tight_layout()

    out_path = os.path.join(base_dir, "mse_matrix_xgb_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    main()
