'''
Classes describing spin models for use with qdccirq.

V1.0 -- new module split off from qdccirq
'''

from typing import Tuple

import numpy as np

import cirq
from openfermion import ops, transforms

from qdclib._quantum_simulated_system import QuantumSimulatedSystem


class TFIMChain(QuantumSimulatedSystem):
    '''
        TODO: TFIMChain documentation
    '''

    def __init__(self, L: int, J: float, B: float, sparse=False,
                 qubits: Tuple[cirq.Qid, ...] = None):
        ''' TODO: TFIMChain.__init__ documentation '''
        self.L = L
        self.J = J
        self.B = B
        if L <= 2 and sparse:
            print(f'! sparse encoding unavailable for {L} qubit system.\n'
                  '! setting self.sparse = False.')
            sparse = False
        ground_state_degeneracy = 2 if J > B else 1
        if qubits is None:
            qubits = [cirq.LineQubit(i) for i in range(L)]
        else:
            qubits = qubits
        self.qubit_ham = ops.QubitOperator('X0', -B)
        for i in range(L - 1):
            self.qubit_ham += ops.QubitOperator(f'X{i+1}', -B)
            self.qubit_ham += ops.QubitOperator(f'Z{i} Z{i+1}', -J)
        sparse_ham = transforms.get_sparse_operator(self.qubit_ham)
        super().__init__(sparse_ham, qubits, ground_state_degeneracy, sparse)

    def get_qubit_hamiltonian(self):
        return self.qubit_ham

    def normalize(self, spread: float = 2.) -> None:
        '''
        Scales and shifts the system Hamiltonian, B, J and shift to achieve
        specified minimum and maximum energies in the Hamiltonian
        eigenspectrum.
        '''
        self.eig()
        previous_spread = self.eigvals[-1] - self.eigvals[0]
        scale = spread / previous_spread
        self.qubit_ham *= scale
        self.sparse_ham *= scale
        self.eigvals *= scale
        self.J *= scale
        self.B *= scale

    def tr2_step(self, dt: float) -> cirq.Circuit:
        '''
        Second-order Trotter step for Hamiltonian simulation.
        '''
        c = cirq.Circuit()
        c.append(
            cirq.XPowGate(exponent=(dt * - self.B / np.pi))(self.qubits[i])
            for i in range(self.L))
        c.append(cirq.ZZPowGate(exponent=(dt * - self.J * 2 / np.pi))(
            self.qubits[2 * i], self.qubits[2 * i + 1])
            for i in range(self.L // 2))
        c.append(cirq.ZZPowGate(exponent=(dt * - self.J * 2 / np.pi))(
            self.qubits[2 * i + 1], self.qubits[2 * i + 2])
            for i in range((self.L - 1) // 2))
        c.append(
            cirq.XPowGate(exponent=(dt * - self.B / np.pi))(self.qubits[i])
            for i in range(self.L))
        return(c)
