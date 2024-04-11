# TANDEM
Tsunami Adjoint simulator for Nonpoint sources with Dispersive wave equations on a self-gravitating Earth Model

# Technical References
- Takagawa, T., Allgeyer, S. & Cummins, P., Adjoint Synthesis for Trans-oceanic Tsunami Waveforms and Simultaneous Inversion of Fault Geometry and Slip Distribution (submitted) 

# How to install
1. Install openmpi
1. Install petsc4py
``` sh
./configure --with-mpi-dir=/usr/local/openmpi-4.0.7 --with-fc=0 --download-f2cblaslapack=1 --download-petsc4py=1
make PETSC_DIR=$HOME/.local/petsc PETSC_ARCH=arch-linux-c-debug all
make PETSC_DIR=$HOME/.local/petsc PETSC_ARCH=arch-linux-c-debug check
```
Add the following line in "~/.bashrc"
``` sh
export PYTHONPATH="$HOME/.local/petsc/arch-linux-c-debug/lib"
```
Then,
``` sh
source ~/.bashrc
python -m pip install git+https://github.com/tomographyyy/tandem.git
```

