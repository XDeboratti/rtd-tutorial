Usage
=====

This section explains how to use the ggnn library and the ggnn module.

Usage in C++
------------

Standard Usage
~~~~~~~~~~~~~~

You can find all the code from this tutorial and additional example files in the :file:`examples/` folder of the GGNN repository.

Before using ggnn, we need to include ``ggnn/base/ggnn.cuh`` from the ggnn library. The header files from the standard library are only for demonstrtaing purposes and are not required for using the library. Then, some data to search in and some data to search the *k*-nearest neighbors for is needed:

.. code:: c++

   #include <ggnn/base/ggnn.cuh>
   #include <array>
   #include <iostream>
   #include <cstdint>
   #include <random>
   using namespace ggnn;

   int main() {

      const size_t N_base = 1000;
      const size_t N_query = 10;
      const uint32_t dim = 123;
   
      //the data to query on
      std::array<float, N_base*dim> base_data;
      //the data to query for
      std::array<float, N_query*dim> query_data;
   
      //generate the data
      std::default_random_engine prng {};
      std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
   
      for(float& x : base_data){
         x = uniform(prng);
      }
      for (float& x : query_data)
         x = uniform(prng);

Then, we  have to initialize a ggnn instance and the datasets:

.. code:: c++

       // data types
       //
       /// data type for addressing points
       using KeyT = int32_t;
       /// data type of the dataset (char, float)
       using BaseT = float;
       /// data type of computed distances
       using ValueT = float;
       using GGNN = GGNN<KeyT, BaseT, ValueT>;
   
      //Initialize ggnn
       GGNN ggnn{};
   
       //Initilaize the datasets containing the base data and query data
       Dataset<BaseT> base = Dataset<BaseT>::copy(base_data, dim, true);
       Dataset<BaseT> query = Dataset<BaseT>::copy(query_data, dim, true);

Instead of copying the data, data on the host can also be referenced with ``referenceCPUData()`` and data on the GPU can be referenced with ``referenceGPUData()``.
If the data is a dataset in fvecs or bvecs format it can be loaded with ``Dataset<BaseT>::load(path_to_file)``.

The base has to be passed to ggnn:

.. code:: c++

       ggnn.setBaseReference(base);

Now, ggnn is ready to be used:

.. code:: c++

       //buid the kNN graph
       ggnn.build(24, 0.5);
       //call query and store indices & squared distances
       const uint32_t KQuery = 10;
       const auto [indices, dists] = ggnn.query(query, KQuery, 0.5);
   
       //print the results for the first query
       std::cout << "Result for the first query verctor: \n";
       for(uint32_t i=0; i < KQuery; i++){
           //std::cout << "Base Idx: ";
           std::cout << "Distance to vector at base[";
           std::cout.width(5);
           std::cout << indices[i];
           std::cout << "]: " << dists[i] << "\n";
       }
      return 0;
   }

``ggnn.build(KBuild, tau_build)`` builds the kNN graph. ``KBuild`` is typically ``24`` and ``tau_build`` is typically ``0 < tau < 2``. In most cases lower numbers are sufficient. ``ggnn.query(query, KQuery, tau_query)`` executes the search. ``query`` is the data to search the *k*-nearest neighbors for. ``KQuery > 0`` can be chosen freely, depending on your needs. ``tau_query`` is again typically ``0 < tau < 2``. However, to finetune performance for your usecase you should play around with those parameters. Refer to the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`Search Parameters <Search_Parameters>` section for more information about parameters and some examples.

Usage with Data on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following the data is assumed to be on the GPU:

.. code:: c++

   #include <ggnn/base/ggnn.cuh>
   #include <ggnn/base/eval.h>
   
   #include <cstdint>
   
   #include <iostream>
   
   #include <cuda_runtime.h>
   #include <curand.h>
   
   using namespace ggnn;
   int main() {
   
       using GGNN = ggnn::GGNN<int32_t, float, float>;
   
       //create data on gpu
       size_t N_base {100000};
       size_t N_query {10000};
       uint32_t D {128};
   
       float* base;
       float* query;
   
       cudaMalloc(&base, N_base*D*sizeof(float));
       cudaMalloc(&query, N_query*D*sizeof(float));
   
       curandGenerator_t generator;
       curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
   
       curandGenerateUniform(generator, base, N_base*D);
       curandGenerateUniform(generator, query, N_query*D);

GGNN has to be initialized but the data can be referenced:

.. code:: c++

   //initialize ggnn
   GGNN ggnn{};
   //set the data on gpu as base on which the graph should be built on, uses a reference to already existing data
   //needs number of base vectors N_base, dimensionality of base vectors D and the gpu_id of the gpu where the data is
   uint32_t gpu_id = 0:
   ggnn.setBase(ggnn::Dataset<float>::referenceGPUData(base, N_base, D, gpu_id));
   //reference the query data which already exists on the gpu
   ggnn::Dataset<float> d_query = ggnn::Dataset<float>::referenceGPUData(query, N_query, D, gpu_id);

Now, ggnn is usable:

.. code:: c++

      //buid the kNN graph
      const uint32_t KBuild = 24;
      const float tau_build = 0.5f;
      ggnn.build(KBuild, tau_build);

      //call query and store indices & distances
      const int32_t KQuery = 10;
      const auto [indices, dists] = ggnn.query(d_query, KQuery, 0.5);
   
      //print the results for the first query
      std::cout << "Result for the first query verctor: \n";
      for(uint32_t i=0; i < KQuery; i++){
         //std::cout << "Base Idx: ";
         std::cout << "Distance to vector at base[";
         std::cout.width(5);
         std::cout << indices[i];
         std::cout << "]: " << dists[i] << "\n";
      }
   
      //cleanup
      curandDestroyGenerator(generator);
      cudaFree(base);
      cudaFree(query);
   
      return 0;
   }




Usage in Python
---------------

You can find all the code from this tutorial and additional example code in the :file:`examples/` folder of the GGNN repository.

Standard Usage
~~~~~~~~~~~~~~

First, we have to import the ggnn module and create the data. Additionally we tell the ggnn module to print the deatiled logs into the console (optional):

.. code:: python

   #! /usr/bin/python3
   
   import ggnn
   import torch
   
   #get detailed logs (optional)
   ggnn.set_log_level(4)
   
   #initialize data
   base = torch.rand((100000, 128), dtype=torch.float32, device='cpu')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cpu')


Then, we have to create an instance of the GGNN class and build the graph. ``build(K_Build, tau_build)`` requires ``K_Build`` and ``tau_build`` as parameters. Typically, ``0 < tau_build < 2``. However, we recommend experimenting with those parameters. Refer to the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`Search Parameters <Search_Parameters>` section for more information about parameters and some examples:

.. code:: python

   #initialize ggnn
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   
   #build the graph
   my_ggnn.build(64, 0.9)

Now, we can query the graph with the created queries and perform a bruteforce query to compare with:

.. code:: python

   #set k_query
   k_query: int = 10

   #run query
   indices, dists = my_ggnn.query(query, k_query, 0.9, 1000)
   
   #run bruteforce query to get a groundtruth and evaluate the results of the query
   indices_eval, dists_eval = my_ggnn.bf_query(query, k_query)
   evaluator = ggnn.Evaluator(base, query, indices_eval, k_query)
   print(evaluator.evaluate_results(indices))

``query(query, k_query, tau_query, max_iterations)`` requires ``query`` (data to query for), ``k_query`` (the number of neighbors to search), ``tau_query`` and ``max_iterations``. To finetune performance for your usecase you should play around with those parameters. Refer to the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`Search Parameters <Search_Parameters>` section for more information about parameters and some examples. The ``Evaluator`` class holds the necessary information for evluating the results of the query. the function ``evaluate_results(indices)`` compares the results of the query (``indices``) with the results from the bruteforce query (``indices_eval``). 

We can also look at the indices of the *k*-nearest neighbors for the first five queries and their squared euclidean distance:

.. code:: python

   print('indices:', indices[:5], '\n dists:',  dists[:5], '\n')

Usage with Data on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

This  works just like with data on the host, you just have to change the device of your torch tensors to ``device='cuda'`` and potentially add the respective gpu index. 

.. code:: python

   #initialize data
   base = torch.rand((100000, 128), dtype=torch.float32, device='cuda')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cuda')


Usage Multi-GPU
~~~~~~~~~~~~~~~

For multi-gpu mode it is required to use ``set_shard_size(N_shard)``, where ``num_base_vectors = N_shard * num gpus`` has to be evenly divisible. Also the GPU ids have to be provided via ``set_gpus()``, which expects an ndarray of GPU ids.

.. code:: python

   #! /usr/bin/python3

   import ggnn
   import torch
   import numpy as np
   
   k_query: int = 10
   
   #initialize data
   base = torch.rand((1000000, 128), dtype=torch.float32, device='cuda')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cuda')
   
   #initialize ggnn and prepare multi gpu
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.set_shard_size(500000)
   my_ggnn.set_gpus(np.array([0,1], dtype=int32))
   
   #build the graph
   my_ggnn.build(64, 0.9)
   
   #run query
   indices, dists = my_ggnn.query(query, k_query, 0.9, 1000)
   
   print('indices:', indices[:5], '\n dists:',  dists[:5], '\n')

.. note::
   The ``Evaluator`` class is only available in single-gpu mode.
