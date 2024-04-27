import numpy as np


def smoothing_U(U, window_size = 3):

    smoothed_U = U.copy().astype(np.float64)

    for i in range(8):
        tmp_U = np.convolve(U[i, :], np.ones(window_size) / window_size, mode="valid")
        smoothed_U[i, window_size//2 : -window_size//2 + 1] = tmp_U
        

    return smoothed_U

