'''
Simulate standard LogSweep cooling (from maximally-mixed state) and reheating
(from ground state) on a normalized TFIM chain system, using density matrix
simulation. Save results to a file.

The output consists in two .json files
'''
import os
import sys

if len(sys.argv) < 6:
    raise(Exception(f"usage: python {sys.argv[0]}"
                    "<L> <J/B> <K> <n_samples> <data_dir>"))
L = int(sys.argv[1])
JvB = float(sys.argv[2])
K = int(sys.argv[3])
n_samples = int(sys.argv[4])
data_dir = sys.argv[5]

import numpy as np
import time

from cirq import Simulator
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
simulator = Simulator()
print('done in', time.time() - stopwatch, 'seconds.')


# cooling
if not cooling_out_exists:
    print('\nRunning cooling simulation')
    stopwatch = time.time()

    norm_samples = []
    energy_samples = []
    gsf_samples = []
    for _ in range(n_samples):
        # initialize random computational-basis system state (this emulates the
        # maximally-mixed system state) tensored with fridge |0>
        init_state = np.zeros(2**(L + 1), dtype=np.complex64)
        init_state[np.random.randint(L)] = 1

        res = simulator.simulate(
            circuit, initial_state=init_state,
            qubit_order=[*system.get_qubits(), qdccirq.FRIDGE]
        )
        final_state = res.final_state[::2]  # discard fridge

        # Compute the norm to normalize sampled results:
        #    The numerical error produces a non-normalized final state.
        #    This seems to be the dominant error in the final result.
        #    To keep this under check, we store the resulting state norms.
        norm = np.sqrt(np.sum(np.abs(final_state)**2))
        norm_samples.append(np.float64(
            norm
        ))  # conversion to np.float64 needed for compatibility with json.
        energy_samples.append(system.energy_expval(final_state) / norm)
        gsf_samples.append(system.ground_space_fidelity(final_state) / norm)

    out_dict = dict(
        L=L,
        J=system.J,
        B=system.B,
        K=K,
        init_state='random computational-basis state',
        energy_avg=np.mean(energy_samples),
        energy_std=np.std(energy_samples),
        gsf_avg=np.mean(gsf_samples),
        gsf_std=np.std(gsf_samples),
        norm_check_samples=norm_samples,
        energy_samples=energy_samples,
        gsf_samples=gsf_samples
    )

    with open(outfile_cooling, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')


# reheating
if not reheating_out_exists:
    print('\nRunning reheating simulation')
    stopwatch = time.time()

    energy_samples = []
    gsf_samples = []
    for _ in range(n_samples):
        # initialize a random ground state of system tensored with fridge |0>
        randidx = np.random.randint(system.ground_state_degeneracy)
        init_state = np.kron(system.eig()[1][:, randidx], [1, 0])

        res = simulator.simulate(
            circuit, initial_state=init_state,
            qubit_order=[*system.get_qubits(), qdccirq.FRIDGE]
        )
        final_state = res.final_state[::2]  # discard fridge

        # Compute the norm to normalize sampled results:
        #    The numerical error produces a non-normalized final state.
        #    This seems to be the dominant error in the final result.
        #    To keep this under check, we store the resulting state norms.
        norm = np.sqrt(np.sum(np.abs(final_state)**2))
        norm_samples.append(np.float64(
            norm
        ))  # conversion to np.float64 needed for compatibility with json.

        energy_samples.append(system.energy_expval(final_state) / norm)
        gsf_samples.append(system.ground_space_fidelity(final_state) / norm)

    out_dict = dict(
        L=L,
        J=system.J,
        B=system.B,
        K=K,
        init_state='random computational-basis state',
        energy_avg=np.mean(energy_samples),
        energy_std=np.std(energy_samples),
        gsf_avg=np.mean(gsf_samples),
        gsf_std=np.std(gsf_samples),
        norm_check_samples=norm_samples,
        energy_samples=energy_samples,
        gsf_samples=gsf_samples
    )

    with open(outfile_reheating, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')
