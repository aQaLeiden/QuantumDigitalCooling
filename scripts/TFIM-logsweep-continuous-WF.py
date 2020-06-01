'''
Simulate standard LogSweep cooling (from maximally-mixed state) and reheating
(from ground state) on a normalized TFIM chain system, using
continuous-evolution wavefunction simulation + Monte Carlo sampling of
nonunitary operations.

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

import json

from qdclib import qdcnumpy
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

system = spinmodels.TFIMChain(L, JvB, 1)
system.normalize()


# cooling
if not cooling_out_exists:
    print('\nRunning cooling simulation')
    stopwatch = time.time()

    norm_samples = []
    energy_samples = []
    gsf_samples = []

    init_state_matr = np.zeros((2**(L), n_samples), dtype=np.complex64)
    for sample in range(n_samples):
        init_state_matr[np.random.randint(L)] = 1

    final_state_matr = qdcnumpy.continuous_logsweep_protocol(
        init_state=init_state_matr,
        system=system,
        n_energy_steps=K,
        mode='WF'
    )

    for final_state in final_state_matr.T:
        norm_samples.append(np.float64(
            np.sqrt(np.sum(np.abs(final_state)**2))
        ))  # conversion to np.float64 needed for compatibility with json.
        energy_samples.append(system.energy_expval(final_state))
        gsf_samples.append(system.ground_space_fidelity(final_state))

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
        energy_samples=energy_samples,
        gsf_samples=gsf_samples,
        norm_samples=norm_samples
    )

    with open(outfile_cooling, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')


# reheating
if not reheating_out_exists:
    print('\nRunning reheating simulation')
    stopwatch = time.time()

    norm_samples = []
    energy_samples = []
    gsf_samples = []

    # construct matrix of column-vectors by repeating GS-subspace basisvectors.
    init_state_matr = np.repeat(
        system.eig()[1][:, :system.ground_state_degeneracy],
        repeats=int(np.ceil(n_samples / system.ground_state_degeneracy)),
        axis=1
    )[:, :n_samples]

    final_state_matr = qdcnumpy.continuous_logsweep_protocol(
        init_state=init_state_matr,
        system=system,
        n_energy_steps=K,
        mode='WF'
    )

    for final_state in final_state_matr.T:
        norm_samples.append(np.float64(
            np.sqrt(np.sum(np.abs(final_state)**2))
        ))  # conversion to np.float64 needed for compatibility with json.
        energy_samples.append(system.energy_expval(final_state))
        gsf_samples.append(system.ground_space_fidelity(final_state))

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
        energy_samples=energy_samples,
        gsf_samples=gsf_samples,
        norm_samples=norm_samples
    )

    with open(outfile_reheating, 'wt') as f:
        json.dump(out_dict, f)

    print('done in', time.time() - stopwatch, 'seconds.')
