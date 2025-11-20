import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(data, seq_len=30):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
