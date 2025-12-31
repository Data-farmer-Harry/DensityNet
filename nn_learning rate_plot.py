import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main() -> None:
    base_dir = os.path.dirname(__file__)
    excel_path = os.path.join(base_dir, "nnerro.xlsx")
    df = pd.read_excel(excel_path)

    required_cols = {"learning_rate", "test_mse"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required_cols)} in {excel_path}")

    df = df.sort_values("learning_rate")
    x = df["learning_rate"].to_numpy()
    y = df["test_mse"].to_numpy()

    out_dir = os.path.join(base_dir, "figure")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nn_learning_rate_plot.png")

    plt.figure(figsize=(4.0, 3.0), dpi=300)
    # plt.plot(x, y, linewidth=1.0, alpha=0.6)
    plt.scatter(x, y, s=10, alpha=0.7, edgecolors="none")
    # plt.yscale("log")
    plt.ylabel("Test MSE")
    plt.xlabel("Learning rate")
    plt.ylabel("Test MSE ")
    plt.title("NN learning rate vs test MSE")
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
