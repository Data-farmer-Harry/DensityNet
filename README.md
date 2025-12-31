# density_net

## Overview
This project builds machine learning models to predict density profiles from MD output.
It supports a feed-forward neural network and an XGBoost regressor, plus scripts to
evaluate errors and generate plots for single cases and summary heatmaps.

## What you are doing here
- Parse raw density files into a single training dataset with width and temperature as inputs.
- Train NN and XGBoost regressors to predict density vs Coord1 in different temperature and width.
- Visualize predictions, errors, and learning-rate sensitivity.
- Export per-(width, temperature) MSE tables and heatmaps.

## Data layout
Raw files live in `density_results/` and follow this filename pattern:
`density_<width>_<temperature>_<anything>.txt`

Example: `density_10_270_100.txt`
- `width` is the first number after `density_`
- `temperature` is the second number
- the last number is ignored by the loader

Each file has a header plus a table:
```
Chunk  Coord1  Ncount  density/mass
```
Only the second column (Coord1) and the last column (density/mass) are used.

## Dataset format
`sample.py` creates `data/train.npz`, `data/valid.npz`, `data/test.npz`.
Each row contains 7 columns:
1) `Coord1`
2) `width`
3) `temperature`


### Train/valid/test split
The split is based on temperature and width:
- Validation: lowest 10% temps + highest 10% temps
- Test: middle temps, then filtered by middle widths
- Train: everything else

## Project structure
- `model.py` - defines the `densityNet`, dataset wrapper, and helpers.
- `sample.py` - build `data/*.npz` from `density_results/*.txt`.
- `train_nn.py` - train the NN once and save `train/NN/model.pth`.
- `nn_training_mse.py` - sweep learning rates and write `nnerro.xlsx`.
- `train_xgboost.py` - XGBoost training with randomized search; saves model and runs a learning-rate sweep to `xgboosterror.xlsx`.
- `plot.py` - per-(width, temperature) curve plot: data vs XGBoost vs NN.
- `plot_error.py` - XGBoost curve plus absolute error subplot.
- `nine_picture.py` - 3x3 panel plot for selected widths/temperatures with error strip.
- `MSE.py` - compute per-(width, temperature) MSE matrix for XGBoost.
- `MSE/Heatmap.py` - render the MSE matrix as a heatmap.
- `nn_learning rate_plot.py` - plot learning rate vs MSE from `nnerro.xlsx`.
- `figure/` - output plots.
- `train/` - trained models and metadata.

## What each figure means
- `figure/test_figure/*.png` from `plot.py`:
  - x-axis: Coord1
  - y-axis: density
  - lines: data (true), XGBoost, NN
- `figure/error/*.png` from `plot_error.py`:
  - top: XGBoost vs data
  - bottom: absolute error vs Coord1
- `figure/nine_3x3.pdf` from `nine_picture.py`:
  - 3 widths x 3 temperatures
  - top: XGBoost vs data
  - bottom: absolute error strip
- `figure/nn_learning_rate_plot.png` from `nn_learning rate_plot.py`:
  - learning rate vs test MSE for the NN
- `MSE/mse_matrix_xgb_heatmap.png` from `MSE/Heatmap.py`:
  - heatmap of XGBoost MSE across width (rows) and temperature (columns)

## Requirements
- Python 3.9+ recommended
- numpy, pandas, matplotlib
- torch
- xgboost
- scikit-learn

## Quick start (demo)
1) Build dataset from raw files:
```
python sample.py
```

2) Train the NN:
```
python train_nn.py
```

3) Train XGBoost and run learning-rate sweep:
```
python train_xgboost.py
```

4) NN learning-rate sweep and plot:
```
python nn_training_mse.py
python "nn_learning rate_plot.py"
```

5) Plot predictions and errors:
```
python plot.py
python plot_error.py
python nine_picture.py
```

6) MSE matrix and heatmap:
```
python MSE.py
python MSE/Heatmap.py
```

## Outputs
- `data/train.npz`, `data/valid.npz`, `data/test.npz`
- `train/NN/model.pth`
- `train/XGBoost/xgb_model.json`
- `nnerro.xlsx` (NN learning-rate sweep)
- `xgboosterror.xlsx` (XGBoost learning-rate sweep)
- plots under `figure/` and `MSE/`

## Notes
- Normalization is computed from `data/train.npz` and used for NN training/inference.
- `nine_picture.py` trims long zero tails after the first non-zero region.
- XGBoost uses GPU if your build supports CUDA (`device="cuda"`).
