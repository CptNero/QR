import numpy as np
from pytictoc import TicToc

from householder import qr_step


def gen_test_data():
    test_data = []

    for i in range(0, 10):
        for n in range(2, 5):
            for m in range(2, 5):
                test_matrix = np.random.random_sample((n, m))
                test_data.append(test_matrix)

    for i in range(0, 10):
        for n in range(2, 5):
            for m in range(2, 5):
                test_matrix = np.random.randint(1, 10, (n, m))
                test_data.append(test_matrix)

    for i in range(0, 10):
        for n in range(2, 5):
            for m in range(2, 5):
                test_matrix = np.random.randint(1, 1000, (n, m))
                test_data.append(test_matrix)

    return test_data


def givens_rotation(a, b):
    c = a / np.sqrt(a * a + b * b)
    s = b / np.sqrt(a * a + b * b)

    if abs(a) > abs(b):
        t = b / a
        c = 1 / np.sqrt(1 + t * t)
        s = c * t
    if abs(b) >= abs(a):
        tau = a / b
        c = s * tau

    return s, c


def GM(p, q, st, ct, n, m):
    shorter_side = max(n, m)
    G = np.eye(shorter_side, shorter_side)
    G[p][p] = ct
    G[q][q] = ct
    G[p][q] = -st
    G[q][p] = st

    return G


def GQR(A):
    n = A.shape[0]
    m = A.shape[1]
    R = A.copy()
    Q = np.eye(n, m)

    for idx in range(0, m):
        for jdx in range(idx + 1, n):
            x = R[idx][idx]
            y = R[jdx][idx]
            st, ct = givens_rotation(x, y)
            G = GM(jdx, jdx - 1, st, ct, n, m)
            R = G @ R
            Q = Q @ G.T

    return Q, R


def HQR(A):
    n = A.shape[0]
    m = A.shape[1]
    Q = np.identity(n)

    R = A.astype(np.float32)
    for i in range(min(n, m)):
        Q, R = qr_step(Q, R, i, n)

    Q = np.around(Q, decimals=6)
    R = np.around(R, decimals=6)

    return Q, R


if __name__ == '__main__':

    Am = np.array([
        [0, 4, 3],
        [-1, 2, 4],
        [1, 0, 0],
    ])

    Am2 = np.array([
        [0, -1, 1],
        [4, 2, 0],
        [3, 4, 0],
    ])

    GmQ, GmR = GQR(Am2)



    t = TicToc()

    householder_time = []
    givens_time = []
    numpy_time = []

    householder_norms = []
    givens_norms = []
    numpy_norms = []

    test_matrices = gen_test_data()

    for matrix in test_matrices:
        t.tic()
        HQ, HR = HQR(matrix)
        householder_time.append(t.tocvalue())
        householder_norms += np.linalg.norm((HQ @ HR) - matrix, 'fro')

        t.tic()
        GQ, GR = GQR(matrix)
        givens_time.append(t.tocvalue())
        givens_norms += np.linalg.norm((GQ @ GR) - matrix, 'fro')

        t.tic()
        NPQ, NPR = np.linalg.qr(matrix)
        numpy_time.append(t.tocvalue())
        numpy_norms += np.linalg.norm((NPQ @ NPR) - matrix, 'fro')
