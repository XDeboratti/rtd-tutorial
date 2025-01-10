Usage
=====

This section explains how to use the ggnn library and the ggnn module.

Usage in C++
------------

Before using ggnn, some data to search in and some data to search the *k*-nearest neighbors for is needed:

.. code::

   #include <ggnn/base/ggnn.cuh>
   #include <array>
   #include <iostream>
   #include <cstdint>
   #include <random>
   using namespace ggnn;

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


      


Usage in Python
---------------
