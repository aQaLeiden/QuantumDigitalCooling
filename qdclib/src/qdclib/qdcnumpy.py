'''
Quantum Digital Cooling (QDC) implementation with numpy and openfermion.

In the continuous QDC steps implemented here, the coupled system-fridge
evolution is simulated analogically:
    U = Exp[-i (H_sys + H_fridge + H_coupl) t]

The module offers two approaches to simulation:
    - with `mode = 'DM'` the state is stored in density matrices, and the
        non-unitary evolution is deterministically implemented
    - with `mode = 'WF'` a number of sample state vectors is stored as column
        vecors in a matrix. The non-unitary part of the evolution is realized
        through Monte Carlo sampling.
'''

from typing import Union, Sequence

import numpy as np
import scipy.linalg

from openfermion import ops, transforms

from ._quantum_simulated_system import QuantumSimulatedSystem
from .qdcutils import (trace_out, perp_norm, opt_delta_factor, logsweep_params)


# **********************************
# *** INTERNAL UTILITY FUNCTIONS ***
# **********************************

def _check_input_set_mode(init_state, mode, dimension):
    '''
    Checks wether init_state has the right form, and sets the 'auto' mode
    to the correct one.

    Args:
        init_state: density matrix, state vector or matrix of state
            coulumn-vectors of the initial state.
        mode: 'auto', 'DM' or 'WF'
        dimension: dimension of system hilbert space (to check input size)

    Returns:
        mode: either 'DM', 'WF-matrix' or 'WF-single'

    Raises:
        ValueError: if init_state is invalid
        TypeError: if the input state is not an array
    '''
    if not isinstance(init_state, np.ndarray):
        raise TypeError('init_state is not a np.ndarray')
    shape = init_state.shape
    if len(shape) not in [1, 2]:
        raise ValueError(f'Invalid shape of input state: {shape}')
    if shape[0] != dimension:
        raise TypeError(f'Invalid shape {shape} for d={dimension}-dimensional '
                        'system Hilbert space')
    if mode == 'DM':
        if not len(shape == 2):
            raise TypeError('Input is not a density matrix and mode is DM')
        if not (shape[0] == shape[1]):
            raise TypeError('Input is not a density matrix and mode is DM')
    if mode == 'auto':
        if len(shape) == 2:
            if shape[0] == shape[1]:
                mode = 'DM'
            else:
                mode = 'WF'
        else:
            mode = 'WF'
    else:
        mode = 'WF'
    if mode == 'WF':
        if len(shape) == 1:
            mode = 'WF-single'
        else:
            mode = 'WF-matrix'
    if mode not in ['DM', 'WF-single', 'WF-matrix']:
        raise ValueError(f'Invalid mode {mode}')
    return mode


# ****************************
# *** CONTINUOUS QDC STEPS ***
# ****************************

def _nonunitary_step_wf_matr(wf_matrix_sys, coupled_evo):
    '''
    TODO implement and test
    '''
    # initialize fridge, tensor it with system state
    full_wf_matr = np.kron(wf_matrix_sys, [[1], [0]])
    # evolve
    full_wf_matr = coupled_evo @ full_wf_matr

    # trace out fridge, which will be in |0> after last reset
    buffer_wf_matr = np.empty_like(wf_matrix_sys)
    for idx in range(wf_matrix_sys.shape[1]):
        # unnormalized system states when fridge is in |0> and |1> respectively
        system_wf_0 = full_wf_matr[::2, idx]
        system_wf_1 = full_wf_matr[1::2, idx]
        norm_0 = np.sum(np.abs(system_wf_0)**2)
        norm_1 = np.sum(np.abs(system_wf_1)**2)
        if np.random.choice([True, False],
                            p=np.array([norm_0, norm_1] / (norm_0 + norm_1))):
            buffer_wf_matr[:, idx] = system_wf_0 / np.sqrt(norm_0)
        else:
            buffer_wf_matr[:, idx] = system_wf_1 / np.sqrt(norm_1)
    # TODO is any optimization possible in this block?

    return buffer_wf_matr


def _nonunitary_step_dm(dm_sys, coupled_evo):
    '''
    TODO test
    '''
    # initialize fridge, tensor it with system state
    dm = np.kron(dm_sys, [[1, 0], [0, 0]])

    # evolve the composite state
    dm = coupled_evo @ dm @ coupled_evo.conjugate().transpose()

    # trace out fridge and return
    return trace_out(dm, -1)


def continuous_evolution_step(
    state: np.ndarray,
    ham_sys: np.ndarray,
    coupling: np.ndarray,
    epsilon: float,
    t: float,
    gamma: float,
    mode: str = 'auto'
) -> np.ndarray:
    '''
    Apply a consinuous evolution + instantaneous reset QDC step to the
    input state.

    Depending on the value of `mode`, either density matrix simulation or
    wavefunction + Monte Carlo sampling for nonunitary operations is used.
    Wavefunction simulation can run in parallel on multiple initial state
    vectors: in this case, init_state should be a matrix which column vectors
    are the state vectors.

    Args:
        init_state: density matrix, state vector or matrix of state
            coulumn-vectors of the initial state.
        ham_sys: Hamiltonian matrix of the system
        coupling: Coupling potential matrix
        epsilon: fridge energy
        t: coupling time
        gamma: coupling strength
        mode: Whether to use density matrix or wavefunction simulation. If set
            to 'auto' (default), the choice will be made depending on the input
            type (square matrix -> 'DM', vector or rectangular matrix -> 'WF')
    Returns:
        System state after application of the step, matching the input type.

    Raises:
        ValueError: if the input state has the wrong form for the mode.
        TypeError: if the input state is not a np.ndarray

    ---

    Typical parameters for continuous QDC steps:
        weak coupling:
            t >> E_transition
            epsilon = E_transition
            gamma = pi / t
        continuous strong coupling:
            t = sqrt(3) pi / (2 E_transition)
            epsilon = E_transition
            gamma = 2/sqrt(3) * E_transition
    '''
    mode = _check_input_set_mode(state, mode, np.shape(ham_sys)[0])

    # compute coupled evolution operator
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    # fast hermitian matrix exponential -> coupled evolution
    eigvals, eigvecs = scipy.linalg.eigh(
        np.kron(ham_sys, np.eye(2))
        + np.kron(np.eye(*np.shape(ham_sys)), - Z / 2 * epsilon)
        + np.kron(coupling, X / 2 * gamma)
    )
    phases = np.exp(1j * eigvals * t)
    evo = eigvecs @ np.diag(phases) @ eigvecs.conjugate().T
    # TODO optimize evo calculation removing np.diag

    # run the nonunitary step corresponding to the selected mode
    if mode == 'DM':
        state = _nonunitary_step_dm(state, evo)
    else:
        if mode == 'WF-single':
            state = state.reshape((-1, 1))
        state = _nonunitary_step_wf_matr(state, evo)
        if mode == 'WF-single':
            state = state.reshape(state.shape[0])
    return state


def continuous_weakcoupling_step(
    state: np.ndarray,
    ham_sys: np.ndarray,
    coupling: np.ndarray,
    epsilon: float,
    delta: float,
    mode: str = 'auto'
) -> np.ndarray:
    '''
    Implementation of the continuous evolution weak coupling step, with
    parameters:
        t = 1 / delta
        gamma = pi * delta
    See continuous_evolution_step for details.
    '''
    return continuous_evolution_step(state, ham_sys, coupling, epsilon,
                                     t=1 / delta, gamma=np.pi * delta)


# ************************
# *** HYBRID QDC STEPS ***
# ************************

# Hybrid steps combine a continuous simulation of system Hamiltonian and a
# trotterized approach to system-fridge coupling simulation.
# TODO


# **************************
# *** FULL QDC PROTOCOLS ***
# **************************

def continuous_energy_sweep_protocol(
    init_state: np.ndarray,
    system: QuantumSimulatedSystem,
    couplings: Union[Sequence[str], Sequence[np.ndarray]],
    epsilon_list: Sequence[float],
    delta_list: Sequence[float],
    mode: str = 'auto'
) -> np.ndarray:
    '''
    Returns the state after application of a generic continuous-evolution
    QDC protocol with chosen fridge energies, linewidths and couplings on a
    QuantumSimulatedSystem.

    Depending on the value of `mode`, either density matrix simulation or
    wavefunction+Monte Carlo sampling for nonunitary operations is used.
    Wavefunction simulation can run in parallel on multiple initial state
    vectors: in which case, init_state should be a matrix which column vectors
    are the state vectors.

    Args:
        init_state: density matrix, state vector or matrix of state
            coulumn-vectors of the initial state.
        system: object containing Hamiltonian and simulation data.
        couplings: how to choose couplings in subsequent cooling steps.
            Accepted choices are:
            - a tuple of values among "X", "Y" and "Z", indicating to choose PX
                couplings for each of the system qubits in sequence, with
                values of P chosen in the tuple.
            - a tuple of matrices representing coupling potentials
        epsilon_list: fridge energy sequence.
        delta_list: cooling linewidth sequence.
        mode: Whether to use density matrix or wavefunction simulation. If set
            to 'auto' (default), the choice will be made depending on the input
            type (square matrix -> 'DM', vector or rectangular matrix -> 'WF')

    Raises:
        ValueError: if the input state has the wrong form for the mode,
            or if the input system is sparse-encoded
        TypeError: if the input state is not a np.ndarray
    '''
    if system.sparse:
        raise ValueError('continuous-evolution simulation does not support'
                         'sparse-encoded sysyems.')
    system.eig()

    if couplings[0] in ['X', 'Y', 'Z']:
        L = len(system.get_qubits())
        couplings = [
            transforms.get_sparse_operator(ops.QubitOperator(f'{P}{qubit}'),
                                           n_qubits=L
                                           ).A
            for qubit in range(L)
            for P in couplings
        ]

    hamiltonian = system.get_sparse_hamiltonian().A
    state = init_state

    for epsilon, delta in zip(epsilon_list, delta_list):
        for coupling in couplings:
            state = continuous_weakcoupling_step(state=state,
                                                 ham_sys=hamiltonian,
                                                 coupling=coupling,
                                                 epsilon=epsilon,
                                                 delta=delta,
                                                 mode=mode)

    return state


def continuous_logsweep_protocol(
    init_state: np.ndarray,
    system: QuantumSimulatedSystem,
    n_energy_steps: int,
    mode: str = 'auto'
) -> np.ndarray:
    '''
    Runs the continuous-evolution QDC LogSweep protocol on the state init_state
    of a QuantumSimulatedSystem.
    For details and caveats, see `qdcnumpy.continuous_energy_sweep_protocol`.

    Couplings are XX, XY, XZ on each of the system's qubits in sequence.
    Energies and linewidths are chosen according to:
        epsilon[0] == e_max
        epsilon[n_energy_steps] == e_min
        epsilon[k] - epsilon[k+1] == (delta[k] + delta[k+1]) / delta_factor
    Where delta_factor is optimized as indicated in appendix B of the paper.

    Args:
        init_state: density matrix, state vector or matrix of state
            coulumn-vectors of the initial state.
        system: object containing Hamiltonian and simulation data.
        n_energy_steps: energy gradation number K.
        mode: Whether to use density matrix or wavefunction simulation. If set
            to 'auto' (default), the choice will be made depending on the input
            type (square matrix -> 'DM', vector or rectangular matrix -> 'WF')

    Returns:
        state (density matrix) after the application of the protocol

    Raises:
        ValueError: if the input state has the wrong form for the mode,
            or if the input system is sparse-encoded
        TypeError: if the input state is not a np.ndarray
    '''
    L = len(system.get_qubits())
    coupl_potentials = [
        transforms.get_sparse_operator(ops.QubitOperator((i, P),), n_qubits=L)
        for i in range(L) for P in ('X', 'Y', 'Z')
    ]
    e_max_transitions = max(perp_norm(cp, system.get_sparse_hamiltonian())
                            for cp in coupl_potentials)
    system.eig()

    #  define e_min, e_max using perp_norm on sparse matrices
    e_min = system.ground_state_gap()
    e_max = e_max_transitions

    # define delta_factor
    delta_factor = opt_delta_factor(e_min, e_max, n_energy_steps)

    epsilon_list, delta_list = logsweep_params(e_min, e_max, n_energy_steps,
                                               delta_factor)

    return continuous_energy_sweep_protocol(init_state=init_state,
                                            system=system,
                                            couplings=['X', 'Y', 'Z'],
                                            epsilon_list=epsilon_list,
                                            delta_list=delta_list,
                                            mode=mode)

#
#
#
#
#