import numpy as np

def index2onehot(n_labels, index):
    return np.eye(n_labels)[index]