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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import dataset, DataLoader

# writer = SummaryWriter(log_dir='train/log/toy')
# data = pd.read_csv('data/train.csv', header=None, index_col=None).to_numpy()

def main():
    data = np.load('data/train.npz')['data']

    # ---- Normalization
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
            data_norm, valid_ratio = 0.00,
            randomSeed = 2022
    )
    test_raw = np.load('data/test.npz')['data']
    test_feat = test_raw[:, :-1]
    test_label = test_raw[:, -1]
    test_feat_norm = (test_feat - FEAT_MEAN) / FEAT_STD
    test_label_norm = (test_label - LABEL_MEAN) / LABEL_STD
    test_data = np.concatenate([test_feat_norm, test_label_norm[:, None]], axis=1)
    print(len(test_data))
    print('Test size =', len(test_data))

    # last column is label
    train_dataset = DensityDataset(data = train_data)
    valid_dataset = DensityDataset(data = valid_data)
    test_dataset = DensityDataset(data = test_data)

    if torch.cuda.is_available():
        device = 'cuda'
        pin_memory = True
        num_workers = 8
    else:
        device = 'cpu'
        pin_memory = False
        num_workers = 0

    batch_size = 8192
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    criterion = nn.MSELoss()
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    model = densityNet(n_in=6, activation='ReLU').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    model_checkpoints_folder = os.path.join('train/', 'NN')
    os.makedirs(model_checkpoints_folder, exist_ok=True)

    n_epochs=100
    log_every = 500
    n_iter = 0
    valid_n_iter = 0
    best_valid_loss = np.inf
    best_valid_mae = np.inf
    best_valid_roc_auc = 0

    for epoch_counter in range(n_epochs):
        model.train()
        for bn, (inputs, target) in enumerate(train_loader):

            input_var = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            optimizer.zero_grad()
            output = model(input_var)

            loss = criterion(output, target)

            if bn % log_every == 0:
                # writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                print('Epoch: %d, Batch: %d, Loss:'%(epoch_counter+1, bn), loss.item())
            loss.backward()
            optimizer.step()
            n_iter += 1
        scheduler.step()

        # validate the model if requested
        if epoch_counter % 1 == 0:
            losses = AverageMeter()
            mae_errors = AverageMeter()
            with torch.no_grad():
                model.eval()
                for bn, (inputs, target) in enumerate(valid_loader):
                    input_var = inputs.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    # compute output
                    output = model(input_var)
                    loss = criterion(output, target)
                    # MAE
                    output_denorm = output * LABEL_STD + LABEL_MEAN
                    target_denorm = target * LABEL_STD + LABEL_MEAN
                    mae_error = mae(output_denorm, target_denorm)
                    mae_errors.update(mae_error, target.size(0))

                print('Epoch [{0}] Validate: [{1}/{2}], '
                        'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch_counter+1, bn+1, len(valid_loader),
                    mae_errors=mae_errors))

            if mae_errors.avg < best_valid_mae:
                # save the model weights
                best_valid_mae = mae_errors.avg
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

            valid_n_iter += 1

    # test the model
    state_dict = torch.load(os.path.join(model_checkpoints_folder, 'model.pth'), map_location=device)
    model.load_state_dict(state_dict)

    mae_errors = AverageMeter()
    with torch.no_grad():
        model.eval()
        for bn, (inputs, target) in enumerate(test_loader):
            input_var = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            output = model(input_var)
            #  MAE
            output_denorm = output * LABEL_STD + LABEL_MEAN
            target_denorm = target * LABEL_STD + LABEL_MEAN
            mae_error = mae(output_denorm, target_denorm)
            mae_errors.update(mae_error, target.size(0))

        print('Test: MAE: {mae_errors.avg:.3f}'.format(
            mae_errors=mae_errors))


    # Plot a regression scatter
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        for inputs, target in test_loader:
            input_var = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input_var)
            output_denorm = (output * LABEL_STD + LABEL_MEAN).cpu().numpy().reshape(-1)
            target_denorm = (target * LABEL_STD + LABEL_MEAN).cpu().numpy().reshape(-1)
            y_pred.append(output_denorm)
            y_true.append(target_denorm)

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

    #  0-100 
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


if __name__ == "__main__":
    main()
