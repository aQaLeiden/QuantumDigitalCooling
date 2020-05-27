'''
Quantum Digital Cooling (QDC) implementation with cirq and openfermion.

V1.0 -- incompatible with previous (unnumbered) module
'''

from typing import Union, Callable, Tuple

import numpy as np

from cirq import (Qid,
                  NamedQubit,
                  Circuit,
                  X, Y, Z, reset,
                  PauliString,
                  ZPowGate)
from openfermion import ops, transforms

from qdclib._quantum_simulated_system import QuantumSimulatedSystem
from qdclib.qdcutils import *

FRIDGE = NamedQubit('fridge')


# ****************************
# *** QDC STEPS AND CYCLES ***
# ****************************

def qdc_step(HS_step_f: Callable[[float], Circuit],
             coupl_step_f: Callable[[float, float], Circuit],
             epsilon: float,
             gamma: float,
             t: float,
             qdc_trotter_number: int,
             HS_trotter_factor: int = 1) -> Circuit:
    '''
    Circuit implementing a generic 2nd-order trotterized QDC step.
    Typical params of QDC steps are indicated below.

    Args:
        HS_step_f (`function(dt)`): function returning cirq.Circuit for a
        2nd order trotter step of the system evolution.
        coupl_step_f (`function(gamma, dt)`): function returning the
            cirq.Circuit for a 1st oder trotter step of evolution genetated by
            the system-fridge coupling.
        epsilon: fridge energy
        gamma: coupling strength
        t: coupling time
        qdc_trotter_number: steps for coupling+hamiltonian 2nd order Trotter
            simulation
        HS_trotter_factor: multiplicative factor for number of Trotter steps
            employed for the system evolution Hamiltonian simulation.

    Returns:
        Circuit of the cooling step. Includes final fridge reset.

    ---

    Typical parameters:
        weak coupling:
            epsilon = E_transition
            t >> E_transition
            gamma = pi / t
            trotter_steps >= 2 sqrt(1 + (E_transition + E_max)^2 /
                                        (2 * gamma)^2 ) 
        bang-bang (Ramsey) strong coupling:
            epsilon = e_transition
            t = pi / (2 * E_transition)
            gamma = 2 e_transition
            trotter_steps = 1
    '''
    dt = t / qdc_trotter_number
    coupl_halfstep = coupl_step_f(gamma, dt / 2)
    c = Circuit([
        coupl_halfstep,
        [HS_step_f(dt / HS_trotter_factor)] * HS_trotter_factor,
        ZPowGate(exponent=(- dt * epsilon / np.pi))(FRIDGE),
        coupl_halfstep,
    ]) * qdc_trotter_number
    c.append(reset(FRIDGE))
    return c


def bangbang_step(HS_step_f: Callable[[float], Circuit],
                  coupl_step_f: Callable[[float, float], Circuit],
                  epsilon: float,
                  HS_trotter_factor: int = 1) -> Circuit:
    '''
    Circuit implementing bang-bang (Ramsey) strong coupling QDC step, with
    2nd-order trotterized system evolution.

    Args:
        HS_step_f (`function(dt)`): function returning the cirq.Circuit for
                a 2nd oder trotter step of the system evolution.
        coupl_step_f (`function(gamma, dt)`): function returning the
                cirq.Circuit for a 1st oder trotter step of evolution
                genetated by the system-fridge coupling.
        epsilon: fridge energy
        HS_trotter_factor: multiplicative factor for number  of trotter steps
                employed for the system evolution Hamiltonian simulation.

    Returns:
        Circuit of the cooling step. Includes final fridge reset.
    '''
    return qdc_step(HS_step_f=HS_step_f,
                    coupl_step_f=coupl_step_f,
                    epsilon=epsilon,
                    gamma=2 * epsilon,
                    t=np.pi / 2 / epsilon,
                    qdc_trotter_number=1,
                    HS_trotter_factor=HS_trotter_factor)


def weakcoupling_step(HS_step_f: Callable[[float], Circuit],
                      coupl_step_f: Callable[[float, float], Circuit],
                      epsilon: float,
                      delta: float,
                      spectral_spread: float,
                      trotter_factor: float = 2.,
                      HS_trotter_factor: int = 1) -> Circuit:
    '''
    Circuit implementing weak-coupling QDC step, with 2nd-order trotterized
    system evolution.

    The number of trotter steps M is calculated as:
        int(trotter_factor * np.sqrt(1 + (E_max) ** 2 / (PI * delta) ** 2))

    Args:
        HS_step_f (`function(dt)`): function returning the cirq.Circuit for
                a 2nd oder trotter step of the system evolution.
        coupl_step_f (`function(gamma, dt)`): function returning the
                cirq.Circuit for a 1st oder trotter step of evolution
                genetated by the system-fridge coupling.
        epsilon: fridge energy
        delta: linewidth (defined as 1/t, the inverse of coupling time)
        E_max: spectral spread of the system (used to compute trotter
            steps)
        HS_trotter_factor: multiplicative factor for number  of trotter steps
            employed for the system evolution Hamiltonian simulation.

    Returns:
        Circuit of the cooling step. Includes final fridge reset.
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

    return qdc_step(HS_step_f=HS_step_f,
                    coupl_step_f=coupl_step_f,
                    epsilon=epsilon,
                    gamma=np.pi * delta,
                    t=1 / delta,
                    qdc_trotter_number=qdc_trotter_number,
                    HS_trotter_factor=HS_trotter_factor)


# **************************
# *** COUPLING FUNCTIONS ***
# **************************

# pauli_gate_exponent = time * splitting / np.pi

def PX_coupl_f(qubit: Qid, P: str) -> Circuit:
    '''
    Generate a PX coupling simulation function [f(gamma, dt) -> cirq.Circuit]
    to use in a QDC step. The function returns the circuit that implements
    a rotation around the PX axis, where `P` is a Pauli matrix acting on
    `qubit` and the X is acting on the fridge.

    args:
        qubit: qubit on which the P acts
        P: either 'X', 'Y' or 'Z'

    Returns:
        a function [f(gamma, dt) -> cirq.Circuit] taking timestep and coupling
        strength gamma and returning the coupling circuit.
    '''
    rot_dict = {'X': X, 'Y': Y, 'Z': Z}

    def f(dt, gamma):
        ps = PauliString([rot_dict[P].on(qubit), X.on(FRIDGE)])
        return Circuit(ps**(dt * gamma / np.pi))

    return f


# **************************
# *** FULL QDC PROTOCOLS ***
# **************************

def energy_sweep_protocol(
    system: QuantumSimulatedSystem,
    couplings: Union[Tuple[str, ...],
                     Tuple[Callable[[float, float], Circuit], ...]],
    epsilon_list: Tuple[float, ...],
    delta_list: Tuple[float, ...]
) -> Circuit:
    '''
    Returns circuit of a generic QDC protocol with chosen fridge energies,
    linewidths and couplings on a QuantumSimulatedSystem.

    Args:
        system: object containing Hamiltonian and simulation data.
        couplings: how to choose couplings in subsequent cooling steps.
            Accepted choices are:
            - a tuple of values among "X", "Y" and "Z", indicating to choose PX
                couplings for each of the system qubits in sequence, with
                values of P chosen in the tuple.
            - a tuple of functions [`function(dt, gamma) -> step_circuit`]
                returning the circuit implementing the 1st-order Trotter
                simulation of the coupling Hamiltonian.
        epsilon_list: fridge energy sequence.
        delta_list: cooling linewidth sequence.
    '''
    eigvals, eigvecs = system.eig()
    spectral_spread = eigvals[-1] - eigvals[0]

    if couplings[0] in ['X', 'Y', 'Z']:
        couplings = [PX_coupl_f(qubit, P)
                     for qubit in system.get_qubits()
                     for P in couplings
                     ]

    c = Circuit()
    for epsilon, delta in zip(epsilon_list, delta_list):
        for coupling in couplings:
            c.append(weakcoupling_step(
                HS_step_f=system.tr2_step,
                coupl_step_f=coupling,
                epsilon=epsilon,
                delta=delta,
                spectral_spread=spectral_spread
            ))
    return c


def logsweep_protocol(system: QuantumSimulatedSystem,
                      n_energy_steps: int) -> Circuit:
    '''
    Runs the QDC LogSweep protocol on a QuantumSimulatedSystem.
    Couplings are XX, XY, XZ on each of the system's qubits in sequence.
    Energies and linewidths are chosen according to:
        epsilon[0] == e_max
        epsilon[n_energy_steps] == e_min
        epsilon[k] - epsilon[k+1] == (delta[k] + delta[k+1]) / delta_factor
    Where delta_factor is optimized as indicated in appendix B of the paper.

    Args:
        system: object containing Hamiltonian and simulation data.
        n_energy_steps: energy gradation number K.
    '''
    L = len(system.get_qubits())
    coupl_potentials = [
        transforms.get_sparse_operator(ops.QubitOperator((i, P),), n_qubits=L)
        for i in range(L) for P in ('X', 'Y', 'Z')
    ]
    e_max_transitions = max(perp_norm(cp, system.get_sparse_hamiltonian())
                            for cp in coupl_potentials)
    eigvals, eigvecs = system.eig()

    # define e_min, e_max using perp_norm on sparse matrices
    e_min = system.ground_state_gap()
    e_max = e_max_transitions

    # define delta_factor
    h = e_max / e_min
    R = np.log(h) * ((1 - h) / (2 * (1 + h)) + np.log(2 * h / (1 + h)))
    delta_factor = np.log(n_energy_steps * 8 / R) / 2 / np.pi

    epsilon_list, delta_list = logsweep_params(e_min, e_max, n_energy_steps,
                                               delta_factor)

    return energy_sweep_protocol(system=system,
                                 couplings=['X', 'Y', 'Z'],
                                 epsilon_list=epsilon_list,
                                 delta_list=delta_list)
