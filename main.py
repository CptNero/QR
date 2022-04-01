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
    if b == 0:
        c = np.sign(a)

        if c == 0:
            c = 1.0
        s = 0
        r = abs(a)
    elif a == 0:
        c = 0
        s = np.sign(b)
        r = abs(b)
    elif abs(a) > abs(b):
        t = b / a
        u = np.sign(a) * np.sqrt(1 + t * t)
        c = 1 / u
        s = c * t
        r = a * u
    else:
        t = a / b
        u = np.sign(b) * np.sqrt(1 + t * t)
        s = 1 / u
        c = s * t
        r = b * u

    return c, s


def GM(i, j, c, s):
    return 0


def GQR(A):
    n = A.shape[0]
    m = A.shape[1]
    Q = np.identity(n)
    R = A.copy()

    # for jdx in range(0, n):
    #     for idx in range(m - 1, jdx + 1):
    #         sin, cos = givens_rotation(R[idx - 1][jdx], R[idx][jdx])
    #         G = GM(idx, jdx, sin, cos)
    #         R = np.transpose(G) @ R
    #         Q = Q @ G

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
    t = TicToc()

    householder_time = []
    givens_time = []
    numpy_time = []

    householder_norms = []
    givens_norms = []
    numpy_norms = []

    A_matrix = np.array([
        [0.8147, 0.9058, 0.1270, 0.9134, 0.6324],
        [0.0975, 0.2785, 0.5469, 0.9575, 0.9649],
        [0.1576, 0.9706, 0.9572, 0.4854, 0.8003],
    ])

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
