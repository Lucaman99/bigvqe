"""Faster construction of sparse matrices"""
import numpy as np
import scipy.sparse
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
from openfermion.chem.molecular_data import spinorb_from_spatial


def spinorb(one_elec):
    n_qubits = 2 * one_elec.shape[0]

    one_body_coefficients = np.zeros((n_qubits, n_qubits))

    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            one_body_coefficients[2 * p, 2 * q] = one_elec[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_elec[p, q]
    return one_body_coefficients


def generate_one_elec_sparse(one_elec):
    one_elec_s = spinorb(one_elec)

    size = one_elec_s.shape[0]

    data_arr = []
    row_arr = []
    col_arr = []

    # Generates one electron terms
    for i in range(size):
        for j in range(size):
            coeff1 = one_elec_s[i][j]
            coeff2 = one_elec_s[j][i]

            if any([coeff1, coeff2]) and i <= j:
                data, rows, cols = apply_fermion([i, j], [0, 1], size)

                if i < j:
                    data_arr.append(coeff1 * data)
                    row_arr.append(rows)
                    col_arr.append(cols)

                    data_arr.append(coeff2 * data)
                    row_arr.append(cols)
                    col_arr.append(rows)
                else:
                    data_arr.append(coeff1 * data)
                    row_arr.append(rows)
                    col_arr.append(cols)
    
    if len(data_arr) > 0:
        final_arr = np.concatenate(data_arr)
        final_row = np.concatenate(row_arr)
        final_col = np.concatenate(col_arr)
    else:
        final_arr, final_row, final_col = [], [], []

    return final_arr, final_row, final_col, size


def generate_sparse(one_elec, two_elec):
    one_elec_s, two_elec_s = openfermion.chem.molecular_data.spinorb_from_spatial(one_elec, two_elec)
    two_elec_s = (1 / 2) * two_elec_s

    size = one_elec_s.shape[0]

    data_arr = []
    row_arr = []
    col_arr = []

    # Generates one electron terms
    for i in range(size):
        for j in range(size):
            coeff1 = one_elec_s[i][j]
            coeff2 = one_elec_s[j][i]

            if any([coeff1, coeff2]) and i <= j:
                data, rows, cols = apply_fermion([i, j], [0, 1], size)

                if i < j:
                    data_arr.append(coeff1 * data)
                    row_arr.append(rows)
                    col_arr.append(cols)

                    data_arr.append(coeff2 * data)
                    row_arr.append(cols)
                    col_arr.append(rows)
                else:
                    data_arr.append(coeff1 * data)
                    row_arr.append(rows)
                    col_arr.append(cols)

    for h in range(size):
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    coeff1 = two_elec_s[h][i][j][k]
                    coeff2 = two_elec_s[i][h][j][k]
                    coeff3 = two_elec_s[h][i][k][j]
                    coeff4 = two_elec_s[i][h][k][j]
                    if (any([coeff1, coeff2, coeff3, coeff4])) and h < i and j < k:
                        data, rows, cols = apply_fermion([h, i, j, k], [0, 0, 1, 1], size)
                        val = (coeff1 + coeff4 - coeff2 - coeff3) * data

                        # h, i, j, k and q, p, s, r
                        data_arr.append(val)
                        row_arr.append(rows)
                        col_arr.append(cols)

    if len(data_arr) > 0:
        final_arr = np.concatenate(data_arr)
        final_row = np.concatenate(row_arr)
        final_col = np.concatenate(col_arr)
    else:
        final_arr, final_row, final_col = [], [], []

    return final_arr, final_row, final_col, size


def sparse_H(one_elec, two_elec, const=None):
    final_arr, final_row, final_col, size = generate_sparse(one_elec, two_elec)
    if const is not None:
        return coo_matrix(scipy.sparse.csr_matrix((final_arr, (final_row, final_col)), shape=(2 ** size, 2 ** size)) + const * scipy.sparse.identity(2 ** size))
    else:
        return coo_matrix((final_arr, (final_row, final_col)), shape=(2 ** size, 2 ** size))

def one_elec_sparse(one_elec):
    final_arr, final_row, final_col, size = generate_one_elec_sparse(one_elec) 
    return coo_matrix((final_arr, (final_row, final_col)), shape=(2 ** size, 2 ** size))