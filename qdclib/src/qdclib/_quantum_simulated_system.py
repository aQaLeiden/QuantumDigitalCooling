'''
Base class for Hamiltonian models compatible with QDC protocols.
'''

from typing import Tuple
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import abc

import cirq

class QuantumSimulatedSystem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, sparse_ham, qubits, ground_state_degeneracy, sparse):
        # parameters that need to be defined in the child class __init__
        self.sparse_ham = sparse_ham
        self.qubits = qubits
        self.ground_state_degeneracy = ground_state_degeneracy

        # flag indicating how to deal with diagonalization. If sparse == True,
        # only the first self.ground_state_degeneracy eigenvalues and the
        # last one will be computed.
        self.sparse = sparse

        self.eigvals = None
        self.eigvecs = None

    # Abstract methods
    @abc.abstractmethod
    def tr2_step(self, dt: float) -> cirq.Circuit:
        '''
        Second-order Trotter step for Hamiltonian simulation.
        '''
        pass

    # Inherited methods
    def get_qubits(self):
        return self.qubits

    def get_sparse_hamiltonian(self):
        return self.sparse_ham

    def eig(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Returns (eigenvalues, eigenvectors) of the system Hamiltonian.
        '''
        if self.eigvals is not None:
            return self.eigvals, self.eigvecs

        if self.sparse is True:
            self.eigvals = np.zeros(4)
            self.eigvecs = np.zeros((self.sparse_ham.shape[0], 4),
                                    dtype=complex)
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(self.sparse_ham,
                                                         which='SA', k=3)
            eigvals, eigvecs = zip(*sorted(zip(eigvals, eigvecs.T)))
            self.eigvals[:3] = eigvals
            self.eigvecs[:, :3] = np.array(eigvecs, dtype=complex).T
            self.eigvals[-1], self.eigvecs[:, -1:] = scipy.sparse.linalg.eigsh(
                self.sparse_ham, which='LA', k=1)
            assert all(self.eigvals[i] <= self.eigvals[i + 1]
                       for i in range(len(self.eigvals) - 1)), \
                f'sparse diagonalization results are not sorted {self.eigvals}'
        else:
            self.eigvals, self.eigvecs = scipy.linalg.eigh(self.sparse_ham.A)
        return self.eigvals, self.eigvecs

    def energy_expval(self, state: np.ndarray):
        '''
        Given a wavefunction or density matrix, returns the expectation value
        of the system Hamiltonian.
        '''
        if len(state.shape) is 1:
            return np.real(state.conj() @ self.sparse_ham @ state)
        elif len(state.shape) is 2:
            return np.real(np.trace(state @ self.sparse_ham))
        raise Exception("input is neither a wavefunction nor a density matrix")

    def eigenstate_occupations(self, state: np.ndarray):
        '''
        Given a wavefunction or density matrix, returns the list of overlaps
        with each eigenstate of the system.
        Unavailable if self.sparse == True
        '''
        if self.sparse:
            raise Exception("eigensate occupations navailable"
                            "for sparse-encoded systems")
        if len(state.shape) is 1:
            return np.real(np.abs(state.conj() @ self.eigvecs)**2)
        elif len(state.shape) is 2:
            return np.real([eigvec.conj() @ state @ eigvec
                            for eigvec in self.eigvecs.T])
        raise Exception("input is neither a wavefunction nor a density matrix")

    def ground_state_gap(self):
        self.eig()
        return self.eigvals[self.ground_state_degeneracy] - self.eigvals[0]

    def ground_space_projector(self, normalized=False):
        '''
            Returns a projector on the subspace spanned by the n-degenerate
            ground states of the system (n=`self.ground_state_degeneracy`).
            If `normalized=True` is passed, the projetor is normalized
            generating a density matrix.
        '''
        self.eig()
        proj = np.sum([np.outer(self.eigvecs[:, i], self.eigvecs[:, i].conj())
                       for i in range(self.ground_state_degeneracy)], axis=0)
        if normalized:
            return proj / self.ground_state_degeneracy
        return proj

    def ground_space_fidelity(self, state):
        '''
            Given a wavefunction or density matrix, returns the overlap
            with the subspace spanned by the n-degenerate ground states
            (n=`self.ground_state_degeneracy`).
        '''
        if len(state.shape) is 1:
            return np.real(np.sum(np.abs(
                state.conj() @ self.eigvecs[:, :self.ground_state_degeneracy]
            )**2))
        if len(state.shape) is 2:
            return np.real(sum(np.abs(
                self.eigvecs[:, i]
                @ state
                @ self.eigvecs[:, i]
            ) for i in range(self.ground_state_degeneracy)))
