""" DISTANCE METRICS 
"""

import numpy as np
from scipy.linalg import get_blas_funcs

def avgMerge(score_matrix):
    return score_matrix.mean().mean()

def maxMerge(score_matrix):
    return score_matrix.max().max()

def minMerge(score_matrix):
    return score_matrix.min().min()

# Compare between every row of x1 and every row of x2
def euclidean(x1,x2):
    assert x1.shape[1]==x2.shape[1]
    x2 = x2.transpose()
    x1_norm = np.sum(np.square(x1), axis=1, keepdims=True)
    x2_norm = np.sum(np.square(x2), axis=0, keepdims=True)
    dist = x1_norm + x2_norm - 2*np.dot(x1,x2)
    return dist

# Compare between every row of x1 and every row of x2
def cosineSimilarity(x1,x2):
    assert x1.shape[1]==x2.shape[1]
    epsilon = 1e-10
    x2 = x2.transpose()
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.dot(x1, x2)
    return dist

def cosineSimilarityBLAS(x1, x2):
    assert x1.shape[1]==x2.shape[1]
    epsilon = 1e-10
    x2 = x2.transpose()
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=0, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    gemm = get_blas_funcs("gemm", [x1, x2])
    dist = gemm(1, x1, x2)
    return dist

# Compare between every row of x1 and x2
def euclidean_pair(x1, x2):
    assert x1.shape == x2.shape
    dist = np.sum(np.square(x1-x2), axis=1)
    return dist

# Compare between every row of x1 and x2
def cosine_pair(x1, x2):
    assert x1.shape == x2.shape
    epsilon = 1e-10
    x1_norm = np.sqrt(np.sum(np.square(x1), axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(np.square(x2), axis=1, keepdims=True))
    x1 = x1 / (x1_norm+epsilon)
    x2 = x2 / (x2_norm+epsilon)
    dist = np.sum(x1 * x2, axis=1)
    return dist