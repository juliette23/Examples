import numpy as np
import pandas as pd
def gramschmidt(X):
    """
    Function that performs the Gram-Schmidt method for a given square Matrix. The matrix must have full rank.
    :param X: square matrix represented as 2D-array object (numpy ndarray);
    :return Q: 2D-array (orthogonal matrix) 
    :return R: 2D-array (upper triangle matrix).
    :return Qqual: norm ||In - Q @ Q.T||
    :return QRqual: norm ||Q.T @ X - R||
    """
    m, n = X.shape
    if m != n:
        print('MATRIX IS NOT SQUARE!')
        return None, None, None, None
    # Initialization of the matrices
    R = np.zeros((n,n))
    Q = np.zeros((n,n))
    R[0, 0] = np.linalg.norm(X[:,0])
    Q[:, 0] = X[:,0]/R[0,0]
    for j in range(1, n):
        w = X[:,j]
        for i in range(0,j):
            R[i,j] = np.dot(Q[:,i],X[:,j])
            w = w - R[i,j]* Q[:,i]
        R[j,j] = np.linalg.norm(w)
        Q[:,j] = w/R[j,j]
    Qqual = np.linalg.norm(np.identity(n)- np.matmul(np.transpose(Q),Q))
    QRqual = np.linalg.norm(np.matmul(np.transpose(Q),X)- R)
    return Q, R, Qqual, QRqual
