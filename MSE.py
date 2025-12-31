import os

import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error


def load_data_and_model():
    train_npz = np.load(os.path.join("data", "train.npz"))["data"]
    test_npz = np.load(os.path.join("data", "test.npz"))["data"]
    valid_npz = np.load(os.path.join("data", "valid.npz"))["data"]
    bst = XGBRegressor()
    bst.load_model(os.path.join("train/XGBoost", "xgb_model.json"))

    return train_npz, test_npz,valid_npz, bst


def main():
    train_npz, test_npz, valid_npz, bst = load_data_and_model()

    os.makedirs("figure", exist_ok=True)

# Combine the train and test data, and calculate the MSE for all (width, temperature) values ​​together.
    all_data = np.vstack([train_npz, test_npz,valid_npz])

    # All widths and temperatures that have appeared
    unique_widths = np.unique(all_data[:, 1])
    unique_temps = np.unique(all_data[:, 2])

    # Initialize the MSE matrix; the default value NaN indicates that this cell contains no data.
    mse_matrix = np.full((len(unique_widths), len(unique_temps)), np.nan, dtype=float)

    # Calculate MSE by combining (width, temperature) elements individually
    for i, w in enumerate(unique_widths):
        for j, t in enumerate(unique_temps):
            mask = (np.isclose(all_data[:, 1], w)) & (np.isclose(all_data[:, 2], t))
            data = all_data[mask]
            if data.size == 0:
                continue

            # Sort by Row
            data = data[np.argsort(data[:, 0])]

            X_raw = data[:, :-1]
            y_true = data[:, -1]

            y_pred = bst.predict(X_raw)
            mse = mean_squared_error(y_true, y_pred)
            mse_matrix[i, j] = mse

    # Rows represent width, columns represent temperature
    df = pd.DataFrame(mse_matrix, index=unique_widths, columns=unique_temps)
    df.index.name = "width"

    excel_path = os.path.join("MSE", "mse_matrix_xgb(new).xlsx")
    df.to_excel(excel_path)
    print(f"MSE matrix (XGBoost) saved to {excel_path}")


if __name__ == "__main__":
    main()

