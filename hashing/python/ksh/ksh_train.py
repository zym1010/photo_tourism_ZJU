"""
KSH_train.py implemented by Zhongwen Xu
according to "Supervised Hashing with Kernels" CVPR 2012
and the matlab code provided by the authors
"""

import numpy as np
from scipy.spatial.distance import *

#TODO load data

# anchors are used for approximation of the nearest neighbor structure
X_anchor = X_train[idx_anchor, :]
dist_train = cdist(X_train, X_anchor, 'euclidean')
# sigma is simply selected as the mean of the distance matrix
sigma = np.mean(dist_train)
# RBF kernel is used
K_train = np.exp(dist_train / sigma);
n_train = K_train.shape[0]
# mean of cols
mcols = np.mean(K_train, 0)
# make kernel matrix zeros-centered
K_train = K_train - np.tile(mclos, (n_train, 1))
# generate matrix S0
for i in xrange(n_train):
    for j in xrange(n_train):
        if train_label[i] == train_label[j]:
            S0[i,j] = 1
        else:
            S0[i,j] = -1
# K_train' * K_train
RM = np.dot(K_train.T, K_train)
A1 = np.zeros((n_anchor, n_bits))
flag = np.zeros((1, n_bits))

for i in xrange(nbits):
    if i 
        S = S - np.dot(y, y.T)
    
    LM = np.dot(np.dot(K_train.T, S), K_train)
    # K'RK a = lambda K'K a
    # V: generlized eigenvalue 
    # vl: left eigenvectors
    # vr: right eigenvectors
    (V, vl, vr) = scipy.linalg.eig(LM, RM)
    # take the eigenvector with the max eigenvalue
    max_idx = np.argmax(np.diag(V))
    a = vr[:, max_idx]
    A1[:,i] = a 
    iter_num = 500
    a_star = Gradient_descent(K_train, S, a, iter_num)
    y = np.array(np.dot(K_train, a) > 0,dtype=int)
    idx = np.nonzero( y <= 0 )
    y[idx] = -1
    
    y1 = np.array(np.dot(K_train, a_star) > 0, dtype=int)
    idx = np.nonzero( y1 <= 0 )
    y1[idx] = -1
    
    if np.dot(np.dot(y1.T, S), y1) > np.dot(np.dot(y.T, S), y)
        flag[i] = 1
        A1[:,i] = a_star
        y = y1

Y = np.array(np.dot(A1.T, K_train.T) > 0, dtype=float)
idx = np.nonzero( Y <= 0)
Y[idx] = -1
