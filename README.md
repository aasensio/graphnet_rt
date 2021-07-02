# graphnet_rt
Graphnets for solving radiative transfer problems in stellar atmospheres

## Database generation
    
The generation of the database (in directory `database`) requires the 
installation of the [Lightweaver](https://github.com/Goobley/Lightweaver)
package. It is recommended to create a new `conda` environment to run this experiment. 

    conda create -n graphnet python=3.8
    conda activate graphnet

Clone the `Lightweaver` repository and install it using

    python -m pip install lightweaver

Now install the following packages for running the generation of the database:

    conda install -c conda-forge numpy scipy astropy mpi4py tqdm argparse

Now, you can run the generation of the database by typing:

    mpiexec -n 10 python database.py --n 10000 --freq 1 --out training
    mpiexec -n 10 python database.py --n 500 --freq 1 --out test

This is a computationally heavy procedure that is MPI parallelized. It will generate a
few files containing temperature stratifications, column mass and optical depths, as
well as departure coefficients.

## Graphnet training

The configuration of the Graphnet model is tuned with a configuration file, that
needs to be passed to the training script. An example is given by `conf.dat`, so that
training can be done using:

    python train.py --conf=conf.dat --gpu=0

## Verification

The results of the training can be checked in some cases for the test set with 

    python test.py
    
## Dependencies

You need to install the following packages:

    conda install -c conda-forge sklearn configobj
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

The installation of PyTorch depends on your specific configuration. Check the [webpage](https://pytorch.org/)
for more information.

This implementation of Graphnet depends on the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) package. Check the
webpage for installation, but here you can find the installation for PyTorch 1.9.0
with CUDA 11.1:

    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
    pip install torch-geometric

or do it with `conda`:

    conda install pytorch-geometric -c rusty1s -c conda-forge