"""Create the big.vqe device"""
import pennylane as qml
from pennylane import QubitDevice
import fqe
import openfermion
import functools
import itertools
from string import ascii_letters as ABC
from openfermion import FermionOperator

import numpy as np
from pennylane import QubitDevice, QubitStateVector, BasisState, DeviceError
from pennylane.operation import DiagonalOperation, Channel
from pennylane.wires import Wires


def single_excitation(wires):
    wire1, wire2 = wires[0], wires[1]
    H = FermionOperator(term=str(wire1) + '^ ' + str(wire2), coefficient=1j)
    H += FermionOperator(term=str(wire2) + '^ ' + str(wire1), coefficient=-1j)


def double_excitation(wires):
    wire1, wire2, wire3, wire4 = wires[0], wires[1], wires[2], wires[3]
    H = FermionOperator(term=str(wire1)+'^ '+str(wire2)+'^ '+str(wire3)+' '+str(wire4), coefficient=1j)
    H += FermionOperator(term=str(wire4)+'^ '+str(wire3)+'^ '+str(wire2)+' '+str(wire1), coefficient=-1j)
    return H


op_map = {
        "SingleExcitation" : single_excitation,
        "DoubleExcitation" : double_excitation,
    }


def generate_H(operation, wires):
    """Generates a Hamiltonian"""
    return op_map[operation.name](wires)


class BigVQEDevice(QubitDevice):

    name = 'Big VQE'
    short_name = 'big.vqe'
    pennylane_requires = '0.1.0'
    version = '0.0.1'
    author = 'Jack Ceroni'

    operations = {
        "SingleExcitation",
        "DoubleExcitation",
        "SingleExcitationPlus",
        "DoubleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitationMinus"
    }

    def __init__(self, wires=None, n_elecs=None, n_orbitals=None, spin=None, compile=False):
        super().__init__(wires=wires)
        self.n_elecs = n_elecs
        self.n_orbitals = n_orbitals
        self.spin = spin

        wfn = fqe.Wavefunction([n_elecs, spin, n_orbitals])
        wfn.set_wfn(strategy="hartree-fock")
        self._state = wfn

        # Allows the user to choose whether to compute all Hamiltonians once, or for every execution
        self.compile = compile
        self.compiled = False

    def _apply_operation(self, operation):
        wires = operation.wires
        self._state = self._state.time_evolve(operation.parameters[0], generate_H(operation, wires))

    def apply(self, operations):

        # apply the circuit operations
        for i, operation in enumerate(operations):

            if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(operation.name, self.short_name)
                )

        for operation in operations:
            self._apply_operation(operation)

    def expval(self, H):
        """Returns the expectation value of an OpenFermion Hamiltonian"""
        return self._state.expectationValue(H)


