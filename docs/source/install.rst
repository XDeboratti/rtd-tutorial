Install
=======

Dependencies
------------

The following dependencies are required to install the library:

- a C++20 compiler (GCC or Clang version 10 or higher)
- CUDA toolkit version 12 or higher
- the Nvidia CUDA compiler nvcc

The existence and version of these dependencies can be checked with:

.. code-block:: console

   nvcc --version

and 

.. code-block:: console

   c++ --version

The reason for one of the following errors, may be a missing compiler or a too old version:

- missing CUDA:
   - ``-- The CUDA compiler identification is unknown``
   - ``Failed to detect a default CUDA architecture.``
- outdated GCC/Clang:
   - ``GCC or Clang version 10 or higher required for C++20 support!``

If the compilers are installed, here are possible fixes:

- CUDA
   - ``export PATH=/usr/local/cuda-12.6/bin/:${PATH}`` or
   - ``export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64/:${LD_LIBRARY_PATH}``
- GCC / Clang
   - ``export CC=gcc-10`` or
   - ``export CXX=g++-10`` or
   - ``export CUDAHOSTCXX=g++-10``


Install ggnn Python Module
---------------------------

To install ggnn, first the repository has to be cloned:

.. code-block:: console

   git clone https://github.com/cgtuebingen/ggnn.git

The easiest way to install ggnn is from the folder containing the repository:

.. code-block:: console

   cd ggnn

The ggnn library can then be installed using the package manager pip: 

.. code-block:: console

   pip install .


.. note::
   Automatic installation via ``pip install ggnn`` is under development.


Install ggnn C++ Library
------------------------

To install ggnn, first the repository has to be cloned:

.. code-block:: console

   git clone https://github.com/cgtuebingen/ggnn.git

The easiest way to install ggnn is from the folder containing the repository:

.. code-block:: console

   cd ggnn

The ggnn library can then be built:

.. code-block:: console

   mkdir build
   cd build
   cmake ..
   make -j4



