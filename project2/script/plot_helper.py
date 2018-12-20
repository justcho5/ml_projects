import numpy as np
import pandas as pd

def plot(predictions):
    data = np.zeros((5, 5), dtype=int)
    for rating in range(1, 6):
        correct = len(predictions[predictions[:, 1] == rating])

        #np.where(predictions[:, 1] == predictions[:, 0])
        #print(correct)
        p = []
        for i in range(1, 6):
            x = len(predictions[(predictions[:, 1] == rating) & (predictions[:, 0] == i)])
            data[(rating - 1, i - 1)] = x

    data = np.nan_to_num(data/data.sum(axis=1, keepdims=True))

    x = pd.DataFrame(data=data, index=range(1, 6), columns=range(1, 6))
    x.plot.bar(stacked=True)