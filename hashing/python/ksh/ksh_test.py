"""
ksh_test.py implemented by Zhongwen Xu
according to "Supervised Hashing with Kernels" CVPR 2012
and the matlab code provided by the authors
"""
import cPickle
import numpy as np
from scipy.spatial.distance import *

with open('ksh.pkl', 'rb') as f:
    (Y, A1, X_anchor, mcols, sigma) = cPickle.load(f)
dist_test = cdist(X_test, X_anchor)
K_test = np.exp(dist_test / sigma)
n_test = K_test.shape[0]
K_test = K_test - np.tile(mclos, (n_test, 1))
test_Y = np.array( np.dot(A1.T, K_test.T) > 0, dtype=int)
idx = np.nonzero(test_Y)
test_Y[idx] = -1
