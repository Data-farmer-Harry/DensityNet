import numpy as np

datas = np.load(r"train.npz", allow_pickle=True)
data = datas['data']

np.savetxt(r"train.txt", data, fmt="%.8f")
