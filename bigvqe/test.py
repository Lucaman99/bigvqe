"""Faster construction of sparse matrices"""
import numpy as np
from scipy.sparse import coo_matrix
import openfermion


def dict_ord(x, y):
    return x[1] <= y[1] if x[0] == y[0] else x[0] <= y[0]


def parity(p):
    return sum(
        1 for (x, px) in enumerate(p)
        for (y, py) in enumerate(p)
        if x < y and px > py
    ) % 2 == 0


def a_dagger(l, i):
    sign = 1

    def a_func(rows, cols, data, n):
        return 2 * rows + 1, 2 * cols, sign * data

    return a_func


def a(l, i):
    count = l - i - 1
    sign = (-1) ** (count)

    def a_func(rows, cols, data, n):
        return 2 * rows, 2 * cols + 1, sign * data

    return a_func


def Z(rows, cols, data, n):
    double_r = 2 * rows
    double_c = 2 * cols

    return np.concatenate((double_r, double_r + 1)), np.concatenate((double_c, double_c + 1)), np.concatenate(
        (data, -1 * data))


def I(rows, cols, data, n):
    double_r = 2 * rows
    double_c = 2 * cols

    return np.concatenate((double_r, double_r + 1)), np.concatenate((double_c, double_c + 1)), np.concatenate(
        (data, data))


def n(l, i):
    count = l - i
    sign = (-1) ** (count)

    def n_func(rows, cols, data, s):
        return 2 * rows + 1, 2 * cols + 1, sign * data

    return n_func


def apply_fermion(indices, sequence, size):
    """Generates a sparse representation of a sequence of fermionic operators"""

    sorted_args = np.lexsort((sequence, indices))
    sorted_seq = [(indices[i], sequence[i]) for i in sorted_args]
    parity_val = bool((1 + len(sequence)) % 2)

    rows, cols, data = np.array([0]), np.array([0]), np.array([1])
    prev = 0

    sign_val = (-1) ** int(parity(sorted_args) + 1)
    i = 0

    while i < len(sorted_seq):
        place, operator = sorted_seq[i]
        f = I if parity_val else Z

        for j in range(prev, place):
            rows, cols, data = f(rows, cols, data, j)

        # Checks if it needs to apply number operator rather than creation/annihilation operators
        if i < len(sorted_seq) - 1:
            if sorted_seq[i + 1][0] == place:
                rows, cols, data = n(len(indices), i)(rows, cols, data, place)
                parity_val = not parity_val
                i += 1
            elif operator == 0:
                rows, cols, data = a_dagger(len(indices), i)(rows, cols, data, place)
            elif operator == 1:
                rows, cols, data = a(len(indices), i)(rows, cols, data, place)
        else:
            if operator == 0:
                rows, cols, data = a_dagger(len(indices), i)(rows, cols, data, place)
            elif operator == 1:
                rows, cols, data = a(len(indices), i)(rows, cols, data, place)

        prev = place + 1
        i += 1
        parity_val = not parity_val

    for j in range(prev, size):
        rows, cols, data = I(rows, cols, data, j)

    return sign_val * data, rows, cols

from tqdm.notebook import tqdm

def generate_sparse_H(one_elec, two_elec):

    one_elec_s, two_elec_s = openfermion.chem.molecular_data.spinorb_from_spatial(one_elec, two_elec)
    two_elec_s = (1/2) * two_elec_s

    size = one_elec_s.shape[0]

    data_arr = []
    row_arr = []
    col_arr = []

    # Generates one electron terms
    for i in range(size):
        for j in range(size):
            coeff = one_elec_s[i][j]
            if coeff != 0.0 and i <= j:
                data, rows, cols = apply_fermion([i, j], [0, 1], size)

                data_arr.append(coeff * data)
                row_arr.append(rows)
                col_arr.append(cols)

                if i < j:
                    data_arr.append(coeff * data)
                    row_arr.append(cols)
                    col_arr.append(rows)

    for h in tqdm(range(size)):
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    coeff = two_elec_s[h][i][j][k]
                    if coeff != 0.0 and h < i and j < k:
                        data, rows, cols = apply_fermion([h, i, j, k], [0, 0, 1, 1], size)
                        val = coeff * data

                        # h, i, j, k and q, p, s, r
                        for _ in range(2):
                            data_arr.append(val)
                            row_arr.append(rows)
                            col_arr.append(cols)

                        # i, h, j, k and h, i, k, j
                        neg_val = -1 * val
                        for _ in range(2):
                            data_arr.append(neg_val)
                            row_arr.append(rows)
                            col_arr.append(cols)

    return coo_matrix((np.concatenate(data_arr), (np.concatenate(row_arr), np.concatenate(col_arr))), shape=(2 ** size, 2 ** size))