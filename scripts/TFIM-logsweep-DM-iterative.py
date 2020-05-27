'''
Simulate iterative LogSweep on a normalized TFIM chain system, using density
matrix simulation. Save results to a file.

Iterative LogSweep consists in iterating the LogSweep(n_energy_steps = k)
protocol for all 2 <= k <= K. This allows efficient cooling from higher-energy
transitions without increasing reheating at the last relevant step.

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

from cirq import DensityMatrixSimulator, Circuit
import json

from qdclib import qdccirq
from qdclib import spinmodels

os.makedirs(data_dir, exist_ok=True)
outfile = os.path.join(data_dir, f"L{L}JvB{JvB}K{K}.json")

if os.path.exists(outfile):
    print(outfile, 'exitst already.')
    print('exiting.')
    exit()


print('\nBuilding circuit')
stopwatch = time.time()
system = spinmodels.TFIMChain(L, JvB, 1)
system.normalize()

circuit = Circuit([qdccirq.logsweep_protocol(system, K)
                   for K_step in range(2, K + 1)])
simulator = DensityMatrixSimulator()
print('done in', time.time() - stopwatch, 'seconds.')


print('\nRunning iterative LogSweep cooling simulation')
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
    protocol='Iterative LogSweep (all k<K)',
    L=L,
    J=system.J,
    B=system.B,
    K=K,
    init_state='maximally-mixed state',
    energy=system.energy_expval(final_state),
    eigoccs=list(system.eigenstate_occupations(final_state))
)

with open(outfile, 'wt') as f:
    json.dump(out_dict, f)

print('done in', time.time() - stopwatch, 'seconds.')
