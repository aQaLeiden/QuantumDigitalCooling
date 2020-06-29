from typing import Union

import numpy as np
import scipy.sparse

from openfermion import ops, transforms


def perp_norm(operator: Union[ops.SymbolicOperator,
                              scipy.sparse.spmatrix,
                              np.ndarray],
              hamiltonian=None):
    '''
    Perpendicular norm of an Hermitian operator or of a commutator of
    Hermitian operators.
    The perpendicular norm is the largest off-diagonal element of an operator
    in any orthonormal basis, and is equal to half of the operator's spectral
    spread.

    If only `operator` is specified, the perpendicular norm of `operator`
    is returned.
    If also `hamiltonian` is specified, the perpendicular norm of the
    commutator of the two is returned.

    if both `operator` and `hamiltonian` are specified, they need be of the
    same type.
    '''
    if hamiltonian is not None:
        if isinstance(operator, ops.SymbolicOperator):
            operator = 1.j * (operator * hamiltonian - hamiltonian * operator)
        else:
            operator = 1.j * (operator @ hamiltonian - hamiltonian @ operator)
    if isinstance(operator, ops.SymbolicOperator):
        operator = transforms.get_sparse_operator(operator)
    if isinstance(operator, scipy.sparse.spmatrix):
        if operator.shape == (1, 1):
            print("ATTENTION: perp_norm is returning 0. "
                  "This might mean that operator and hamiltonian commute")
            return 0
        elif operator.shape == (2, 2):
            eigvals = scipy.linalg.eigh(operator.A)[0]
            return (eigvals[1] - eigvals[0]) / 2
        return (scipy.sparse.linalg.eigsh(operator, k=1, which='LA')[0][0]
                - scipy.sparse.linalg.eigsh(operator, k=1,
                                            which='SA')[0][0]) / 2
    eigvals = scipy.linalg.eigvalsh(operator)
    return (eigvals[-1] - eigvals[0]) / 2


def logsweep_params(e_min: float,
                    e_max: float,
                    n_energy_steps: float,
                    delta_factor: int = 1):
    '''
    List of epsilons and deltas to implement LogSweep QDC protocol, defined by:
    epsilon[0] == e_max
    epsilon[n_energy_steps] == e_min
    epsilon[k] - epsilon[k+1] == (delta[k] + delta[k+1]) / delta_factor

    Returns:
        a tuple of lists (epsilon_list, delta_list)

    N.B. delta_factor == 2 / alpha (as defined in appendix A of QDC paper)
    '''
    e_list = np.logspace(np.log10(e_max), np.log10(e_min), n_energy_steps)
    d_list = delta_factor * e_list * \
        (1 - 2 / (1 + (e_min / e_max)**(1 / (1 - n_energy_steps))))
    return e_list, d_list


def opt_delta_factor(e_min: float, e_max: float, n_energy_steps: int) -> float:
    '''
    Returns the optimal delta_factor for the standard LogSweep protocol,
    according to appendix A of the QDC paper
    '''
    h = e_max / e_min
    R = np.log(h) * ((1 - h) / (2 * (1 + h)) + np.log(2 * h / (1 + h)))
    delta_factor = np.log(n_energy_steps * 8 / R) / 2 / np.pi
    return delta_factor


def trotter_number_weakcoupling_step(delta: float,
                                     spectral_spread: float,
                                     trotter_factor: int = 2) -> int:
    '''
    Default number of trotter steps M for a QDC weak coupling step:
        M = int(trotter_factor * np.sqrt(1 + (E_max) ** 2 / (PI * delta) ** 2))
    '''

    # old choice of number of trotter steps:
    # qdc_trotter_number = int(trotter_factor *
    #     np.sqrt( 1 + ( epsilon + E_max ) ** 2 / ( 2 * np.pi * delta ) ** 2 )
    # )

    # new choice of number of trotter steps:
    qdc_trotter_number = int(
        trotter_factor * np.sqrt(1 + (spectral_spread) ** 2
                                 / (np.pi * delta) ** 2)
    )

    return qdc_trotter_number


def trace_out(state: np.ndarray, qubit_index: int) -> np.ndarray:
    '''
        Partial trace on a one qubit-subspace.

        Args:
            state: a [2**N, 2**N] matrix (i.e. N qubit density matrix).
            qubit_index: index in [0, ..., N-1] of the qubit to be traced out

        Returns:
            the [2**(N-1), 2**(N-1)] matrix representing the state with one
                qubit traced out.

        Raises:
            TypeError if the input matrix is not intepretable as a qubit-space
                operator.
    '''
    size = int(np.log2(state.shape[0]))
    if len(np.shape(state)) is not 2:
        raise TypeError(f'state is not a matrix')
    if state.shape[0] != state.shape[1]:
        raise TypeError('state matrix is not square')
    if not (state.shape[0] == 2**size):
        raise TypeError("state's space dimension is not a power of 2")
    if -size <= qubit_index < 0:
        qubit_index = size + qubit_index
    if not (0 <= qubit_index < size):
        raise ValueError(f'invalid qubit_index {qubit_index}')
    cut_indices = (np.arange(2**size) % 2**(size - qubit_index)
                   < 2**(size - qubit_index - 1))
    ncut_indices = np.logical_not(cut_indices)
    return (state[cut_indices, :][:, cut_indices]
            + state[ncut_indices, :][:, ncut_indices])
