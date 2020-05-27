'''
Simulate standard LogSweep cooling (from maximally-mixed state) and reheating
(from ground state) on a normalized TFIM chain system, using density matrix
simulation. Save results to a file.

The output consists in two .json files
'''
import os
import sys

if len(sys.argv) < 4:
    raise(Exception(f"usage: python {sys.argv[0]} <L> <J/B> <K> <data_dir>"))
L = int(sys.argv[1])
JvB = float(sys.argv[2])
K = int(sys.argv[3])
data_dir = sys.argv[4]

import numpy as np
import time

from cirq import DensityMatrixSimulator
import json

from qdclib import qdccirq
from qdclib import spinmodels


cooling_dir = os.path.join(data_dir, "cooling")
reheating_dir = os.path.join(data_dir, "reheating")
os.makedirs(cooling_dir, exist_ok=True)
os.makedirs(reheating_dir, exist_ok=True)
outfile_cooling = os.path.join(cooling_dir, f"L{L}JvB{JvB}K{K}.json")
outfile_reheating = os.path.join(reheating_dir, f"L{L}JvB{JvB}K{K}.json")


reheating_out_exists = False
if os.path.exists(outfile_cooling):
    print(f'output file {outfile_cooling} exists already.')
    cooling_out_exists = True
else:
    cooling_out_exists = False
if os.path.exists(outfile_reheating):
    print(f'output file {outfile_reheating} exists already.')
    reheating_out_exists = True
else:
    reheating_out_exists = False
if cooling_out_exists and reheating_out_exists:
    print('exiting.')
    exit()

print('\nBuilding circuit')
stopwatch = time.time()
system = spinmodels.TFIMChain(L, JvB, 1)
system.normalize()
circuit = qdccirq.logsweep_protocol(system, K)
simulator = DensityMatrixSimulator()
print('done in', time.time() - stopwatch, 'seconds.')


# cooling
if not cooling_out_exists:
    print('\nRunning cooling simulation')
    stopwatch = time.time()

    # build maximally-mixed system state tensored with |0><0| fridge
    init_state = np.diag(
        np.tile(np.array([1 / 2**L, 0], dtype=np.complex64), 2**L))

    res = simulator.simulate(
        circuit, initial_state=init_state,
        qubit_order=[*system.get_qubits(), qdccirq.FRIDGE]
    )
    final_state = res.final_density_matrix[::2, ::2]  # discard fridge

    out_dict = dict(
        L=L,
        J=system.J,
        B=system.B,
        K=K,
        init_state='maximally-mixed state',
        energy=system.energy_expval(final_state),
        eigoccs=list(system.eigenstate_occupations(final_state))
    )

    with open(outfile_cooling, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')


# reheating
if not reheating_out_exists:
    print('\nRunning reheating simulation')
    stopwatch = time.time()
    system_ground_state = system.ground_space_projector(normalized=True)
    init_state = np.array(np.kron(system_ground_state, [[1, 0], [0, 0]]),
                          dtype=np.complex64)
    res = simulator.simulate(
        circuit, initial_state=init_state,
        qubit_order=[*system.get_qubits(), qdccirq.FRIDGE]
    )
    final_state = res.final_density_matrix[::2, ::2]  # discard fridge

    out_dict = dict(
        L=L,
        J=system.J,
        B=system.B,
        K=K,
        init_state='ground state',
        energy=system.energy_expval(final_state),
        eigoccs=list(system.eigenstate_occupations(final_state))
    )

    with open(outfile_reheating, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')
