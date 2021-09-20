"""Base file for fermionic emulator"""
import numpy as np


def factorial(n):
    """Factorial of a positive integer"""
    prod = 1
    for k in range(1, n + 1):
        prod *= k
    return prod


def binomial(n, k):
    """Computes the binomial coefficient"""
    return int(factorial(n) / (factorial(k) * factorial(n - k)))


def swap_bits(n, b1, b2):
    """Returns an integer with bits at position b1 and b2 switched"""


def u():
    """Returns the u1 and u2 matrices for constructing single excitations"""
    pass

class Wavefunction:
    """Initializes an empty fermionic wavefunction"""
    def __init__(self, n_orb, n_elec, spin):
        self.n_orb = n_orb
        self.spin = spin
        self.n_elec = n_elec
        self.dim = binomial(int(self.n_orb/2), int((self.n_elec + self.spin)/2)) * binomial(int(self.n_orb/2), int((self.n_elec - self.spin)/2))

        self.state = None

    def state_set(self, new_state):
        """Sets the matrix representation of the wavefunction"""
        self.state = new_state

    def state_mult(self, U):
        """Contracts a tensor U against the matrix wavefunction"""
        self.state = np.einsum("abcd,cd->ab", U, self.state)

        def apply_fermion(indices, sequence, size):
            """Generates a sparse representation of a sequence of fermionic operators"""

            sorted_args = np.lexsort((sequence, indices))
            sorted_seq = [(indices[i], sequence[i]) for i in sorted_args]
            parity_val = bool(1 + len(sequence) % 2)

            rows, cols, data = np.array([0]), np.array([0]), np.array([1])
            prev = size

            sign_val = (-1) ** int(parity(sorted_args) + 1)
            i = len(sorted_seq) - 1

            while i >= 0:
                place, operator = sorted_seq[i]
                f = I if not parity_val else Z

                for j in range(place + 1, prev):
                    rows, cols, data = f(rows, cols, data, j)

                # Checks if it needs to apply number operator rather than creation/annihilation operators
                if i > 0:
                    if sorted_seq[i - 1][0] == place:
                        rows, cols, data = n(len(indices), i)(rows, cols, data, place)
                        parity_val = not parity_val
                        i -= 1
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
                i -= 1
                parity_val = not parity_val

            f = I if not parity_val else Z

            for j in range(0, place):
                rows, cols, data = f(rows, cols, data, j)

            return coo_matrix((sign_val * data, (rows, cols)), shape=(2 ** size, 2 ** size))