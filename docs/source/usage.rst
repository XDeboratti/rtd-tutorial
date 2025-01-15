Usage
=====

This section explains how to use the ggnn library and the ggnn module.

Usage in C++
------------

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

       //buid the kNN graph, needs KBuild (the number of neighbors each node should have)
       //typically KBuild = 24, in more complex data more neighbors might be usefull and
       //tau_build which controls the stopping criterion for the searches during graph construction
       //typically 0 < tau < 2, lower numbers are sufficient in most cases
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

``ggnn.build(KBuild, tau_build)`` builds the kNN graph. ``KBuild`` is typically ``24`` and ``tau_build`` is typically ``0 < tau < 2``. In most cases lower numbers are sufficient. However, to finetune performance for your usecase you may play around with those two parameters. Refer to `GGNN: Graph-based GPU Nearest Neighbor
Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`Search Parameters <Search_Parameters>` section for more information about those two parameters and some examples.

Usage of ggnn if data is already on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

You can find all the code from this tutorial and additional example code in the :file:`python-src/ggnn/` folder of the GGNN repository.

First, we have to import the module (for that purpose we use sys, you may do that as you please). Torch is only imported to generate data

.. code:: python

   #! /usr/bin/python3
   
   import sys
   sys.path.append('path_to_build_folder')
   import GGNN
   import torch
   
   base = GGNN.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_base.fvecs')
   query = GGNN.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_query.fvecs')
   gt = GGNN.IntDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_groundtruth.ivecs')
   
   k_query: int = 10
   
   evaluator = GGNN.Evaluator(base, query, gt, k_query)
   
   #base = torch.rand((100000, 128), dtype=torch.float32, device='cuda')
   #base = torch.rand((90000, 128), dtype=torch.float32, device='cuda')
   #base = torch.rand((50000, 128), dtype=torch.float32, device='cuda')
   #query = torch.rand((10000, 128), dtype=torch.float32, device='cuda')
   #base = torch.rand((2048, 4096), dtype=torch.float32, device='cuda')
   #query = torch.rand((256, 4096), dtype=torch.float32, device='cuda')
   
   ggnn = GGNN.GGNN()
   ggnn.set_base(base)
   ggnn.build(24, 0.5)
   
   indices, dists = ggnn.query(query, k_query, 0.34, 200)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.34, 400)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.41, 200)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.41, 400)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.51, 200)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.51, 400)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.64, 200)
   print(evaluator.evaluate_results(indices, gt))
   indices, dists = ggnn.query(query, k_query, 0.64, 400)
   print(evaluator.evaluate_results(indices, gt))
