Install
=======

The GGNN library can be installed on Linux by compiling GGNN with cmake and then using the python package manager pip.

Dependencies
------------

Some dependencies are necessary to build the C++ library:

- a C++20 compiler (GCC or Clang version 10 or higher)
- CUDA toolkit version 12
- nvcc

You can check for those dependencies via:

.. code-block:: console

   nvcc --version

and 

.. code-block:: console

   c++ --version

If you receive one of the following errors a missing compiler or too old version might be the reason:

- missing CUDA:
   - ``-- The CUDA compiler identification is unknown``
   - ``Failed to detect a default CUDA architecture.``
- outdated GCC/Clang:
   - ``GCC or Clang version 10 or higher required for C++20 support!``

If you have the compilers installed, potential fixes are:

- CUDA
   - ``export PATH=/usr/local/cuda-12.6/bin/:${PATH}`` or
   - ``export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64/:${LD_LIBRARY_PATH}``
- GCC / Clang
   - ``export CC=gcc-10`` or
   - ``export CXX=g++-10`` or
   - ``export CUDAHOSTCXX=g++-10``


.. _Install_Cpp_Library:

Install GGNN C++ Library
------------------------

To use ggnn, first clone the repository:

.. code-block:: console

   git clone --recursive https://github.com/cgtuebingen/ggnn.git

Navigate to the folder containing the repository:

.. code-block:: console

   cd ggnn

Build the library:

.. code-block:: console

   mkdir build
   cd build
   cmake ..
   make -j4


Install ggnn Python Module
---------------------------

.. note::
   Automatic installation via the python package manager pip is under development.

First, follow the steps in :ref:`Install_Cpp_Library <Install_Cpp_Library>`.

Second, navigate into the ggnn folder:

.. code-block:: console

   cd ggnn

Then, use the package manager pip: 

.. code-block:: console

   pip install .
