This folder contains data generated with an early version of scripts/TFIM-logsweep-WF.py
The samples use the raw output state of cirq.Simulator.simulate, which have the tendency 
to be unnormalized due to the dominant numerical error.
For this reason, we obtain numerical artifacts, like average energies below the ground state energy.

