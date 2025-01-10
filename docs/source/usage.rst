Usage
=====

This section explains how to use the ggnn library and the ggnn module.

Usage in C++
------------

Before using ggnn, some data to search in and some data to search the *k*-nearest neighbors for is needed:

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

Instead of copying the data, data on the host can also be referenced with ``referenceCPUData()`` and data on the GPU can be referenced with :kbd:`referenceGPUData()`.
If the data is a dataset in fvecs or bvecs format it can be loaded with :kbd:`Dataset<BaseT>::load(:file:`path_to_file`)`.

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
