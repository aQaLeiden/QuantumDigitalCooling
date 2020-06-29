'''
Simulate iterative LogSweep on a normalized TFIM chain system, using
wavefunction simulation + Monte Carlo sampling of non-unitary operations.
Save results to a file.

Iterative LogSweep consists in iterating the LogSweep(n_energy_steps = k)
protocol for all 2 <= k <= K. This allows efficient cooling from higher-energy
transitions without increasing reheating at the last relevant step.

The output consists in a .json file
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

from cirq import Circuit, Simulator
import json

from qdclib import qdccirq
from qdclib import spinmodels

# prepare file path
os.makedirs(data_dir, exist_ok=True)
outfile = os.path.join(data_dir, f"L{L}JvB{JvB}K{K}.json")

# Check existance of output to avoid repeating simulations.
if os.path.exists(outfile):
    print(outfile, 'exitst already.')
    print('exiting.')
    exit()

print('\nBuilding circuit')
stopwatch = time.time()
system = spinmodels.TFIMChain(L, JvB, 1)
system.normalize()
circuit = Circuit((qdccirq.logsweep_protocol(system, K_step)
                   for K_step in range(2, K + 1)))
simulator = Simulator()
print('done in', time.time() - stopwatch, 'seconds.')

# iterative
print('\nRunning iterative LogSweep cooling simulation')
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
    energy_samples.append(system.energy_expval(final_state) / norm**2)
    gsf_samples.append(system.ground_space_fidelity(final_state) / norm**2)

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

with open(outfile, 'wt') as f:
    json.dump(out_dict, f)

print('done in', time.time() - stopwatch, 'seconds.')