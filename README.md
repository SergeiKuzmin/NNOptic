NNOptic
====

   This library is a python implementation of a neural network 
for modeling linear optical circuits.
It contains tools that allow you to train a neural network, 
as well as work with an already trained network to find the optimal 
phases required for a given transformation. 
The library itself is located in the nnoptic folder, 
and the examples folder contains typical examples.

Installation
====

**It is recommended** to install the library 
in a virtual environment.

### Linux
```
   python -m venv NNOptic
   source NNOptic/bin/activate
```

### Windows
```
   python -m venv NNOptic
   NNOptic\Scripts\activate.bat 
```

## Installing from source code
To install the stable or development version, 
you need to install from the source. 
First, clone the repository:

```
   git clone https://gitlab.com/SergeiKuzmin/nnoptic.git
```

```
   python setup.py install
```
What those packages do
====

They have the following functionality:

- `nnoptic` : The library itself
- `nnoptic.training` : Contains a class of a neural network and functions for training it
- `nnoptic.tuning` : Contains a class of a interferometer and functions for tuning it
- `nnoptic.functionals` :  Contains functions that compute functionals
- `nnoptic.funcs_for_matrix` : Contains auxiliary functions for working with matrices
- `nnoptic.load_data` : Contains auxiliary functions for creating and loading basis matrices
- `examples` : Contains typical examples that use the nnoptic library

Current maintainer is [Sergei Kuzmin](https://github.com/SergeiKuzmin).
