GGNN Documentation
===================================

The `ggnn library <https://github.com/cgtuebingen/ggnn>`_ provides an efficient algorithm for approximate nearest neighbor (ANN) search that takes advantage of the massive parallelism offered by GPUs. 
It is written in C++ and has Python 3 wrappers, so it can be used from both languages.
The library is based on the method proposed in the paper `GGNN: Graph-based GPU Nearest Neighbor
Search <https://arxiv.org/abs/1912.01059>`_ by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P.A. Lensch.

The :doc:`install` section explains how to install the library, and the :doc:`usage` section provides short tutorials and code examples.

Contents
--------

.. toctree::

   Home <self>
   ann-and-ggnn
   install
   usage
   FAQ

Capabilities and Limitations
----------------------------

The ggnn library supports...

- arbitrarily large datasets, the only limit is your hardware.
- data with up to 4096 dimensions
- building graphs with up to 512 edges per node
- searching for up to 6000 nearest neighbors

.. caution::

   We do not recommended to search for :math:`k = 6000` neighbors, the recommended range is :math:`k \in [1, 1000]`.

