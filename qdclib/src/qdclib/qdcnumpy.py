'''
Quantum Digital Cooling (QDC) implementation with cirq and openfermion.

V1.0 -- incompatible with previous (unnumbered) module
'''

from typing import Union, Tuple

import numpy as np
import scipy.linalg

from openfermion import ops, transforms

from qdclib._quantum_simulated_system import QuantumSimulatedSystem
from qdclib.qdcutils import *


# ****************************
# *** CONTINUOUS QDC STEPS ***
# ****************************

def _cont_evo_step_dm(dm_sys, ham_sys, coupling, epsilon, t, gamma):
    '''
        Internal function: implement QDC step in the case the input is a DM.
    '''
    # initialize fridge, tensor it with system state
    dm = np.kron(dm_sys, [[1, 0], [0, 0]])

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

    # evolve the composite state
    dm = evo @ dm @ evo.conjugate().transpose()

    # trace out fridge and return
    return trace_out(dm, -1)


def _cont_evo_step_wf(wf_sys, ham_sys, coupling, epsilon, t, gamma):
    '''
        Internal function: implement QDC step in the case the input is a WF,
            using Monte Carlo sampling for the reset.
    '''
    # initialize fridge, tensor it with system state
    wf = np.kron(wf_sys, [1, 0])

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

    # evolve the composite state
    wf = evo @ wf

    system_wf_0 = wf[::2]   # unnormalized system state when fridge is in |0>
    system_wf_1 = wf[1::2]  # unnormalized system state when fridge is in |1>
    norm_0 = np.sum(np.abs(system_wf_0)**2)
    norm_1 = np.sum(np.abs(system_wf_1)**2)
    if np.random.choice([True, False], p=[norm_0, norm_1]):
        return system_wf_0 / np.sqrt(norm_0)
    else:
        return system_wf_1 / np.sqrt(norm_0)


def continuous_evolution_step(state: np.ndarray,
                              ham_sys: np.ndarray,
                              coupling: np.ndarray,
                              epsilon: float,
                              t: float,
                              gamma: float) -> np.ndarray:
    '''
        Apply a consinuous evolution + instantaneous reset QDC step to the
        input state.
        Depending on the input, either density matrix simulation or Monte
        Carlo sampling of nonunitary operations is used.
        see below for typical QDC parameters.

        Args:
            state: density matrix or wavefunction encoding the state of the
                system. If a density matrix is given, density matrix (exact)
                simulation is performed. If a vector is given, Monte Carlo
                sampling is used for non-unitary resets.
            ham_sys: Hamiltonian matrix of the system
            coupling: Coupling potential matrix
            epsilon: fridge energy
            t: coupling time
            gamma: coupling strength

        Returns:
            System state after application of the step, as a density matrix or
                wavefunction matching the input type.

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
    if len(np.shape(state)) == 2:
        return _cont_evo_step_dm(state, ham_sys, coupling, epsilon, t, gamma)
    elif len(np.shape(state)) == 1:
        return _cont_evo_step_wf(state, ham_sys, coupling, epsilon, t, gamma)
    else:
        raise TypeError('the input state is not a wavefunction or density'
                        'matrix')


def continuous_weakcoupling_step(state: np.ndarray,
                                 ham_sys: np.ndarray,
                                 coupling: np.ndarray,
                                 epsilon: float,
                                 delta: float) -> np.ndarray:
    '''
        Implementation of the continuous evolution weak coupling step,
        with parameters:
            t = 1 / delta
            gamma = pi * delta
        See continuous_evolution_step for details.
    '''
    return continuous_evolution_step(state, ham_sys, coupling, epsilon,
                                     t=1 / delta, gamma=np.pi * delta)


# ************************
# *** HYBRID QDC STEPS ***
# ************************

# TODO


# **************************
# *** COUPLING FUNCTIONS ***
# **************************


# **************************
# *** FULL QDC PROTOCOLS ***
# **************************

def _continuous_energy_sweep_protocol_dm(
    init_state: np.ndarray,
    system: QuantumSimulatedSystem,
    couplings: Union[Tuple[str, ...],
                     Tuple[np.ndarray, ...]],
    epsilon_list: Tuple[float, ...],
    delta_list: Tuple[float, ...]
) -> np.ndarray:
    '''
    Returns the state after application of a generic QDC protocol with chosen
    fridge energies, linewidths and couplings on a QuantumSimulatedSystem.

    NB: for performance, this function acts directly on init_state. Deepcopy
        input state on function call if side effects have to be avoided.

    Args:
        init_state: density matrix of the initial state.
        system: object containing Hamiltonian and simulation data.
        couplings: how to choose couplings in subsequent cooling steps.
            Accepted choices are:
            - a tuple of values among "X", "Y" and "Z", indicating to choose PX
                couplings for each of the system qubits in sequence, with
                values of P chosen in the tuple.
            - a tuple of matrices representing coupling hamiltonians
        epsilon_list: fridge energy sequence.
        delta_list: cooling linewidth sequence.
    '''
    eigvals, eigvecs = system.eig()

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
                                                 delta=delta
                                                 )

    return state


def continuous_energy_sweep_protocol(
    init_state: np.ndarray,
    system: QuantumSimulatedSystem,
    couplings: Union[Tuple[str, ...],
                     Tuple[np.ndarray, ...]],
    epsilon_list: Tuple[float, ...],
    delta_list: Tuple[float, ...]
) -> np.ndarray:
    '''
    Returns the state after application of a generic continuous-evolution
    QDC protocol with chosen fridge energies, linewidths and couplings on a
    QuantumSimulatedSystem.

    Depending on the input, either density matrix simulation or Monte
    Carlo sampling of nonunitary operations is used.
    (TODO wavefunction approach currently unavailable)

    NB: for performance, this function acts directly on init_state. Deepcopy
        input state on function call if side effects have to be avoided.

    Args:
        init_state: density matrix or wavefunction of the initial state.
        system: object containing Hamiltonian and simulation data.
        couplings: how to choose couplings in subsequent cooling steps.
            Accepted choices are:
            - a tuple of values among "X", "Y" and "Z", indicating to choose PX
                couplings for each of the system qubits in sequence, with
                values of P chosen in the tuple.
            - a tuple of matrices representing coupling potentials
        epsilon_list: fridge energy sequence.
        delta_list: cooling linewidth sequence.
    '''
    if len(np.shape(init_state)) == 2:
        return _continuous_energy_sweep_protocol_dm(
            init_state, system, couplings, epsilon_list, delta_list)
    elif len(np.shape(init_state)) == 1:
        raise Exception('Wavefunction aproach yet not available')
    else:
        raise TypeError('the input state is not a wavefunction or density'
                        'matrix')


def continuous_logsweep_protocol(init_state: np.ndarray,
                                 system: QuantumSimulatedSystem,
                                 n_energy_steps: int) -> np.ndarray:
    '''
    Runs the continuous-evolution QDC LogSweep protocol on the state init_state
    of a QuantumSimulatedSystem.

    Couplings are XX, XY, XZ on each of the system's qubits in sequence.
    Energies and linewidths are chosen according to:
        epsilon[0] == e_max
        epsilon[n_energy_steps] == e_min
        epsilon[k] - epsilon[k+1] == (delta[k] + delta[k+1]) / delta_factor
    Where delta_factor is optimized as indicated in appendix B of the paper.

    Args:
        init_state: density matrix or wavefunction of the initial state.
        system: object containing Hamiltonian and simulation data.
        n_energy_steps: energy gradation number K.

    Returns:
        state (density matrix) after the application of the protocol
    '''
    L = len(system.get_qubits())
    coupl_potentials = [
        transforms.get_sparse_operator(ops.QubitOperator((i, P),), n_qubits=L)
        for i in range(L) for P in ('X', 'Y', 'Z')
    ]
    e_max_transitions = max(perp_norm(cp, system.get_sparse_hamiltonian())
                            for cp in coupl_potentials)
    eigvals, eigvecs = system.eig()

    #  define e_min, e_max using perp_norm on sparse matrices
    e_min = system.ground_state_gap()
    e_max = e_max_transitions

    # define delta_factor
    h = e_max / e_min
    R = np.log(h) * ((1 - h) / (2 * (1 + h)) + np.log(2 * h / (1 + h)))
    delta_factor = np.log(n_energy_steps * 8 / R) / 2 / np.pi

    epsilon_list, delta_list = logsweep_params(e_min, e_max, n_energy_steps,
                                               delta_factor)

    return continuous_energy_sweep_protocol(init_state=init_state,
                                            system=system,
                                            couplings=['X', 'Y', 'Z'],
                                            epsilon_list=epsilon_list,
                                            delta_list=delta_list)


#
#
#
#
#
