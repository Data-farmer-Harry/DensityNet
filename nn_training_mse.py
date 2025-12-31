from model import *
# from utils import binMolDen
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader



data = np.load(r"data\train.npz")["data"]

feat = data[:, :-1]
label = data[:, -1]
FEAT_MEAN = feat.mean(axis=0)
FEAT_STD = feat.std(axis=0)
FEAT_STD[FEAT_STD == 0] = 1.0
LABEL_MEAN = label.mean()
LABEL_STD = label.std()
if LABEL_STD == 0:
    LABEL_STD = 1.0

feat_norm = (feat - FEAT_MEAN) / FEAT_STD
label_norm = (label - LABEL_MEAN) / LABEL_STD
data_norm = np.concatenate([feat_norm, label_norm[:, None]], axis=1)

train_data, valid_data = split_data(
    data_norm,
    valid_ratio=0.00,
    randomSeed=2022,
)

test_raw = np.load(r"data\test.npz")["data"]
test_feat = test_raw[:, :-1]
test_label = test_raw[:, -1]
test_feat_norm = (test_feat - FEAT_MEAN) / FEAT_STD
test_label_norm = (test_label - LABEL_MEAN) / LABEL_STD
test_data = np.concatenate([test_feat_norm, test_label_norm[:, None]], axis=1)
print(len(test_data))
print("Test size =", len(test_data))

train_dataset = DensityDataset(data=train_data)
valid_dataset = DensityDataset(data=valid_data)
test_dataset = DensityDataset(data=test_data)

train_loader = DataLoader(
    train_dataset, batch_size=1024, num_workers=0, drop_last=False, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=1024, num_workers=0, drop_last=False, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=1024, num_workers=0, drop_last=False, shuffle=False
)
 
criterion = nn.MSELoss()
device = "cuda" if torch.cuda.is_available() else "cpu"

model_checkpoints_folder = r"train\NN"
os.makedirs(model_checkpoints_folder, exist_ok=True)


# The program is trained iteratively at different learning rates, and the average error is recorded in nnerro.xlsx
learning_rates = [round(x, 3) for x in np.linspace(0.001, 0.2, 5)]
n_epochs = 40

log_every = 500

results = []
best_overall_mse = float("inf")
best_overall_mae = float("inf")
best_state_dict = None
best_lr = None

for lr in learning_rates:
    print(f"\n==== Training with learning rate = {lr} ====")

    model = densityNet(n_in=6, activation="ReLU").to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    for epoch_counter in range(n_epochs):
        model.train()
        for bn, (inputs, target) in enumerate(train_loader):
            input_var = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(input_var)
            loss = criterion(output, target)

            if bn % log_every == 0:
                print(
                    "LR: {:.5f}, Epoch: {:d}, Batch: {:d}, Loss: {:.6f}".format(
                        lr, epoch_counter + 1, bn, loss.item()
                    )
                )
            loss.backward()
            optimizer.step()

        scheduler.step()

    mae_errors = AverageMeter()
    mse_errors = AverageMeter()
    with torch.no_grad():
        model.eval()
        for bn, (inputs, target) in enumerate(test_loader):
            input_var = inputs.to(device)
            target = target.to(device)

            output = model(input_var)
            output_denorm = output * LABEL_STD + LABEL_MEAN
            target_denorm = target * LABEL_STD + LABEL_MEAN

            mae_error = mae(output_denorm, target_denorm)
            mse_error = F.mse_loss(output_denorm, target_denorm)

            mae_errors.update(float(mae_error), target.size(0))
            mse_errors.update(float(mse_error), target.size(0))

    avg_mae_value = float(mae_errors.avg)
    avg_mse_value = float(mse_errors.avg)
    print(
        "LR {:.5f} -> Test MAE: {:.6f}, Test MSE: {:.6f}".format(
            lr, avg_mae_value, avg_mse_value
        )
    )
    results.append({"learning_rate": lr, "test_mae": avg_mae_value, "test_mse": avg_mse_value})

    # Record the current optimal model
    if avg_mse_value < best_overall_mse:
        best_overall_mse = avg_mse_value
        best_state_dict = model.state_dict()
        best_lr = lr
    if avg_mae_value < best_overall_mae:
        best_overall_mae = avg_mae_value


results_df = pd.DataFrame(results)
results_path = os.path.join("nnerro.xlsx")
results_df.to_excel(results_path, index=False)
print(f"Learning-rate MAE/MSE results saved to {results_path}")
print(
    "Best learning rate: {lr}, Test MAE: {mae:.6f}, Test MSE: {mse:.6f}".format(
        lr=best_lr, mae=best_overall_mae, mse=best_overall_mse
    )
)


# Save and use the model parameters corresponding to the optimal learning rate.
if best_state_dict is not None:
    best_model_path = os.path.join(model_checkpoints_folder, "model.pth")

    torch.save(best_state_dict, best_model_path)
    print(f"Best model saved to {best_model_path}")

    model = densityNet(n_in=6, activation="ReLU").to(device)
    model.load_state_dict(best_state_dict)
else:
    raise RuntimeError("No best model state dict was recorded.")

mae_errors = AverageMeter()
mse_errors = AverageMeter()
with torch.no_grad():
    model.eval()
    y_true = []
    y_pred = []
    for bn, (inputs, target) in enumerate(test_loader):
        input_var = inputs.to(device)
        target = target.to(device)
        output = model(input_var)

        output_denorm = (output * LABEL_STD + LABEL_MEAN).cpu().numpy().reshape(-1)
        target_denorm = (target * LABEL_STD + LABEL_MEAN).cpu().numpy().reshape(-1)

        mae_error = mae(
            torch.tensor(output_denorm), torch.tensor(target_denorm)
        )
        mse_error = F.mse_loss(
            torch.tensor(output_denorm), torch.tensor(target_denorm)
        )
        mae_errors.update(float(mae_error), target_denorm.shape[0])
        mse_errors.update(float(mse_error), target_denorm.shape[0])

        y_pred.append(output_denorm)
        y_true.append(target_denorm)

    print(
        "Best model Test MAE: {:.6f}, Test MSE: {:.6f}".format(
            float(mae_errors.avg), float(mse_errors.avg)
        )
    )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

fig, ax = plt.subplots(figsize=(4, 4), dpi=600)
ax.scatter(y_true, y_pred, s=5, alpha=0.5, color="C0")
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
ax.set_xlabel("True c_myRDF[2]", fontsize=12)
ax.set_ylabel("Predicted c_myRDF[2]", fontsize=12)
ax.set_title("Regression: prediction vs true", fontsize=12)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

# Zoom in on the 0-100 range
low, high = 0.0, 100.0
fig, ax = plt.subplots(figsize=(4, 4), dpi=600)
ax.scatter(y_true, y_pred, s=5, alpha=0.5, color="C0")
ax.plot([low, high], [low, high], "r--", linewidth=1)
ax.set_xlabel("True c_myRDF[2]", fontsize=12)
ax.set_ylabel("Predicted c_myRDF[2]", fontsize=12)
ax.set_title("Regression: prediction vs true (0-100)", fontsize=12)
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()
