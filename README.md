# TANDEM
Tsunami Adjoint simulator for Nonpoint sources with Dispersive wave equations on a self-gravitating Earth Model

## Technical References
- Takagawa, T., Allgeyer, S. & Cummins, P., Adjoint Synthesis for Trans-oceanic Tsunami Waveforms and Simultaneous Inversion of Fault Geometry and Slip Distribution (submitted) 

## How to install
tandem can be installed in the following 7 steps. This is an example for Ubuntu. Windows users are recommended to use WSL (Windows Subsystem for Linux).

### 1. gfortran & pip
```sh
sudo apt update
sudo apt-get update
sudo apt install python3-pip
sudo apt install build-essential
sudo apt install gfortran
```

### 2. Numpy

```sh
python -m pip install --user numpy
```

### 3. FFTW

```sh
cd /tmp
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar xf fftw-3.3.10.tar.gz
cd fftw-3.3.10/
./configure --help # check options
./configure --enable-openmp --enable-shared --enable-avx
make -j8
sudo make install
# fftw3 is installed at /usr/local
sudo nano /etc/ld.so.conf
-----
include /etc/ld.so.conf.d/*.conf # default
/usr/local/lib                   # Add this line
-----
sudo ldconfig # update the library link
ldconfig -p | grep libfftw # check the library link
```

### 4. Shtns

```sh
cd /tmp
wget https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns/-/archive/master/shtns-master.tar.gz
tar xf shtns-master.tar.gz
cd shtns-master/
./configure --help # check options
./configure --enable-openmp --enable-python
make -j8
sudo -E python setup.py install # -E option is needed for super user to find numpy module 
```

### 5. OpenMPI

```sh
cd /tmp
wget "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.gz"
tar xf openmpi-5.0.3.tar.gz
cd openmpi-5.0.3/
./configure --help # check options
./configure CC=gcc CXX=g++ F77=gfortran FC=gfortran # This may take a few minutes.
make -j8 # This may take a few minutes.
sudo make install # lib
sudo ldconfig # update the library link

# check installation
mpicc -v
mpiexec --version
```

### 6. petsc4py

```sh
# install requirements
sudo apt-get install libblas-dev liblapack-dev
python -m pip install --user cython --proxy=10.72.10.106:8080

cd /tmp
wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.0.tar.gz
tar xf petsc-3.21.0.tar.gz
cd petsc-3.21.0/
./configure --help
./configure --download-petsc4py=1

# This may take several minutes.
make -j8 PETSC_DIR=/tmp/petsc-3.21.0 PETSC_ARCH=arch-linux-c-debug all 
# output
=====================================
To use petsc4py, add /tmp/petsc-3.21.0/arch-linux-c-debug/lib to PYTHONPATH
=====================================

nano ~/.bashrc
-----
export PYTHONPATH=/tmp/petsc-3.21.0/arch-linux-c-debug/lib
-----
source ~/.bashrc

sudo make install
```

### 7. tandem

```sh
python -m pip install git+https://github.com/tomographyyy/tandem.git
```
