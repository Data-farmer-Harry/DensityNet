import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold

USE_LOG_Y = False 


def load_data():
    train_npz = np.load(os.path.join("data", "train.npz"))["data"]
    test_npz  = np.load(os.path.join("data", "test.npz"))["data"]

    X_train = train_npz[:, :-1]
    y_train = train_npz[:, -1]
    X_test  = test_npz[:, :-1]
    y_test  = test_npz[:, -1]

    if USE_LOG_Y:
        y_train = np.log1p(y_train)

    return X_train, y_train, X_test, y_test


def assert_gpu_used(model: xgb.XGBRegressor):
    cfg = model.get_booster().save_config().lower()

    gpu_signals = ["\"device\":\"cuda\"", "cuda:", "gpu_hist", "gpu_id", "gputree"]
    if not any(s in cfg for s in gpu_signals):
        raise RuntimeError(
            "GPU-only mode: XGBoost did not report CUDA/GPU in booster config. "
            "This usually means your xgboost is CPU-only or CUDA is not available."
        )


def main():
    X_train, y_train, X_test, y_test = load_data()

    # robust objective 
    # objective = "reg:absoluteerror"
    objective = "reg:squarederror"

    try:
        _ = xgb.XGBRegressor(objective=objective)
    except Exception:
        objective = "reg:pseudohubererror"


    base_model = xgb.XGBRegressor(
        objective=objective,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        random_state=2022,
        n_jobs=1,
        verbosity=2,
    )



    param_distributions = {
        "n_estimators": [1200, 1800, 2500, 3500],
        "learning_rate": [0.005, 0.01, 0.02, 0.03],
        "max_depth": [5, 7, 9, 11],
        "min_child_weight": [0.5, 1, 2, 3],
        "subsample": [0.85, 1.0],
        "colsample_bytree": [0.85, 1.0],
        "gamma": [0.0, 0.05, 0.1],
        "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2],
        "reg_lambda": [0.5, 1.0, 2.0],
        "max_bin": [256, 512, 1024],
        "max_leaves": [0, 63, 127, 255],
    }


    cv = KFold(n_splits=5, shuffle=True, random_state=2022)


    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=80,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=1,
        random_state=2022,
        n_jobs=1,
    )


    print("XGBoost hyper-parameter search (GPU only)")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    assert_gpu_used(best_model)

    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    y_pred = best_model.predict(X_test)
    if USE_LOG_Y:
        y_pred = np.expm1(y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"XGBoost Test MAE (best model): {mae:.4f}")

    out_dir = os.path.join("train", "XGBoost")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "xgb_model.json")
    best_model.save_model(model_path)
    print(f"XGBoost model saved to: {model_path}")

    meta = {"use_log_y": USE_LOG_Y, "objective": objective, "device": "cuda"}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Evaluate different learning rates using the best params from search.
    learning_rates = [round(x, 3) for x in np.linspace(0.001, 0.2, 5)]
    lr_results = []

    for lr in learning_rates:
        params = dict(search.best_params_)
        params["learning_rate"] = lr

        model = xgb.XGBRegressor(
            objective=objective,
            eval_metric="rmse",
            tree_method="hist",
            device="cuda",
            random_state=2022,
            n_jobs=1,
            verbosity=0,
            **params,
        )
        model.fit(X_train, y_train)

        y_pred_lr = model.predict(X_test)
        if USE_LOG_Y:
            y_pred_lr = np.expm1(y_pred_lr)

        mse = mean_squared_error(y_test, y_pred_lr)
        lr_results.append({"learning_rate": lr, "test_mse": float(mse)})
        print(f"LR {lr:.3f} -> Test MSE: {mse:.6f}")

    lr_df = pd.DataFrame(lr_results)
    lr_path = os.path.join("xgboosterror.xlsx")
    lr_df.to_excel(lr_path, index=False)
    print(f"Learning-rate MSE results saved to {lr_path}")


if __name__ == "__main__":
    main()
