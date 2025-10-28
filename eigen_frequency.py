# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 21:41:48 2025

@author: yihao
"""

import numpy as np
from scipy.linalg import eigh

if __name__=="__main__": 
    M = np.load("M.npy")
    K = np.load("K.npy")
    K = K * 200
    M = M * 1e-5
    eigen_values, eigen_vectors = eigh(K, M)
    evals = eigen_values[eigen_values > 1e-8]

    # Natural frequencies
    omegas = np.sqrt(evals)         # rad/s
    freqs = omegas / (2*np.pi)      # Hz
    print(freqs)
    #print(eigen_values)