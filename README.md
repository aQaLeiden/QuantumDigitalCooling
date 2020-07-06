# QuantumDigitalCooling
Simulate quantum digital cooling protocols on model systems.

Quantum Digital Cooling (QDC) is a class of digital qauntum computing methods for approximate ground state or thermal state preparation, introduced in arXiv:1909.10538.
This repository incudes the code used for simulations that will be reported in a new version of arXiv:1909.10538 (work in progress).


## qdclib

The package `qdclib` (included in this repo) needs to be installed to run QDC simulation scripts.
To install it:
- clone this repository and `cd` into its main directory
- eventually activate the environment of choice
- to install in editable mode, run `pip install -e ./qdclib`


## QDC simulations

### Scripts
Scripts to run simulations of standard QDC protocols are incuded in `./scripts/`.
The usage is documented in each script.
All scripts require `qdclib` to be installed (see above).
Scripts save results as a dictionary in a `.json` file, named using the parameters passed to the script, in the specified data folder.

### Data
A set of pre-computed simulation outputs is included in this repository, under `./data`.
The subdirectory structure of data follows the hierarchy:
```
`./data`
└── `TFIM` - simulated system model
    └── `logsweep` - protocol
        └── `DM` - simulator type. eventually more than 1 level (e.g. `continuous/DM/`)
            ├── `cooling` - initial state or other protocol details
            │    └── `K2JvB2L2.json` - data files
            ├── `reheating`
            └── `iterative`
```

### running scripts with SLURM
QDC simulations are lengthy, for this reason they have been run on a HPC cluster [TODO cite maris].
A script to create and instantiate SLURM jobs running simulations is provided `./SLURM-working-directory/SLURM-launcher.ipynb` (for convenience, in a Jupyter notebook where the simulation parameters that have been run can be logged).
This makes sure all needed folders exist, generates SBATCH scripts, and submits them to the queue manager.
