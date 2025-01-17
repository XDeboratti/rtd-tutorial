GGNN Documentation
===================================

The `ggnn library <https://github.com/cgtuebingen/ggnn>` provides an efficient algorithm for approximate nearest neighbor (ANN) search that takes advatage of the massive parallelism offered by GPUs. 
It is written in C++ and has Python 3 wrappers, so it can be used from both languages.
The library is based on the method proposed in the paper `GGNN: Graph-based GPU Nearest Neighbor
Search <https://arxiv.org/abs/1912.01059>`_ by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P.A. Lensch.

The :doc:`install` section shows how to :ref:`install <Install_Cpp_Library>` the project, and the :doc:`usage` section provides a short tutorial and code examples.

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

- arbitrarily large datasets, the only limit is your hardware here.
- data with up to 4096 dimensions
- building graphs with up to 512 edges per node
- searching for up to 6000 nearest neighbors

..caution::

   It is not recommended to search for ``k_query = 6000`` neighbors, we recommend to search for ``k_query <= 1000``.

