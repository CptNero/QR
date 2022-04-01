import numpy as np


def householder_mtx(h):
    H = np.eye(h.shape[0]) - 2 / (h.T.dot(h)) * (h.dot(h.T))
    return H


def householder_transformation(v):
    """
    visszaadja a Householder mátrixot
    """
    size_of_v = v.shape[0]

    # megkonstruáljuk azt a vektort, amely első elemként tartalmazza a v vektor 2-es normáját, utána csupa 0
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = np.linalg.norm(v, 2) * e1

    # a fenti előadásból átemelt mondat szerint, a 2-es norma előjele az v vektor első eleme alapján:
    if v[0, 0] < 0:
        vector = - vector

    u = v + vector.astype(np.float32)

    # Householder mátrix gyártás:
    H = householder_mtx(u)
    return H


def column_converter(x):
    """
    Az oszlopfolytonosan vett elemeket (1d array) oszlopvektorrá konvertáljuk
    """
    return x.reshape(-1, 1)


def qr_step(q, r, iter, n):
    """
    visszaadja a Q es R matrixokat (egy iteráció)
    """
    # oszlopfolytonosan haladunk, pl. r[iter:, iter] az első oszlop lesz ha iter = 0
    v = column_converter(r[iter:, iter])
    # ez alapján létrehozzuk a Householder mátrixot
    Hbar = householder_transformation(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    # itt lehetne @, vagy H.dot(r) is
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r
