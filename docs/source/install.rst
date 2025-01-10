Install
=======

The GGNN library can be installed on Linux by compiling GGNN with cmake and then using the python package manager pip.

.. _Install_Cpp_Library:

Install C++ Library
-------------------

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
   make


Install Python Module
---------------------

First, follow the steps in :ref:`Install_Cpp_Library <Install_Cpp_Library>`.

Then, use the package manager pip: 

.. code-block:: console

   pip install ggnn


