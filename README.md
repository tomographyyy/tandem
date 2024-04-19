# TANDEM


**T**sunami **A**djoint simulator for **N**onpoint sources using **D**ispersive wave equations on a self-gravitating **E**arth **M**odel  


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10995292.svg)](https://doi.org/10.5281/zenodo.10995292)


![Fig01](https://github.com/tomographyyy/tandem/assets/34155315/95713702-778b-476f-a0bf-944c79ae1aa9)

  
![model_difference](https://github.com/tomographyyy/tandem/assets/34155315/ec8608e8-7cea-4471-a1dd-0e8afd444b7f)


## Technical References
- Takagawa, T., Allgeyer, S. & Cummins, P., Adjoint Synthesis for Trans-oceanic Tsunami Waveforms and Simultaneous Inversion of Fault Geometry and Slip Distribution (submitted) 

## How to install
TANDEM can be installed in the following 7 steps. This is an example for Ubuntu. Windows users are recommended to use WSL (Windows Subsystem for Linux).

#### 1. gfortran & pip
```sh
sudo apt update
sudo apt-get update
sudo apt install python3-pip
sudo apt install build-essential
sudo apt install gfortran
```

#### 2. Numpy

```sh
python -m pip install --user numpy
```

#### 3. FFTW

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

#### 4. Shtns

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

#### 5. OpenMPI & mpi4py

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

# mpi4py
python -m pip install mpi4py
```

#### 6. petsc & petsc4py

```sh
# install requirements
sudo apt-get install libblas-dev liblapack-dev
python -m pip install --user cython --proxy=10.72.10.106:8080

cd /tmp
wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.0.tar.gz
tar xf petsc-3.21.0.tar.gz
cd petsc-3.21.0/
./configure --help
sudo mkdir /opt/petsc
sudo chown $USER /opt/petsc
./configure --prefix=/opt/petsc --download-hypre --download-petsc4py=1

# This may take several minutes.
make -j8 PETSC_DIR=/tmp/petsc-3.21.0 PETSC_ARCH=arch-linux-c-debug all
make PETSC_DIR=/tmp/petsc-3.21.0 PETSC_ARCH=arch-linux-c-debug install
# output
=====================================
To use petsc4py, add /opt/petsc/lib to PYTHONPATH
=====================================

nano ~/.bashrc
-----
export PYTHONPATH=/opt/petsc/lib
-----
source ~/.bashrc

```

#### 7. TANDEM

```sh
python -m pip install git+https://github.com/tomographyyy/tandem.git
```

## Run TANDEM

```sh
mpiexec -n 8 python tandem/run.py
```

## Visualize outputs

- Example ipynb is provided. See "view_results.ipynb"
