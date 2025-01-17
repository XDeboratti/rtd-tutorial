Usage
=====

This section explains how to use the ggnn module and the ggnn library. The ggnn library is written for easy use in Python, but of course using it from C++ is also explained here.


Usage in Python
---------------

The code from this tutorial and additional examples can be found in the :file:`ggnn/examples/ggnn_pytorch.py` file of the GGNN repository.

Standard Usage
~~~~~~~~~~~~~~

After installing the ggnn module, it needs to be imported and example data needs to be created. The dimensionality of the data has to be :math:`d >= 1`. ``ggnn.set_log_level(4)`` prints log information into the console during the execution of the algorithm, the higher the log level, the more information, 0 is the lowest log level and if no log level is set, the log level is automatically 0:

.. code:: python

   #! /usr/bin/python3
   
   import ggnn
   import torch
   
   #get detailed logs
   ggnn.set_log_level(4)
   
   
   #create data
   base = torch.rand((100000, 128), dtype=torch.float32, device='cpu')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cpu')


The next step is to create an instance of the GGNN class from the ggnn module. The GGNN class needs the base data (``my_ggnn.set_base(base)``) and can build the graph:

.. code:: python

   #initialize ggnn
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   
   #build the graph
   my_ggnn.build(k_build=64, tau_build=0.9)

The parameters of the ``build(k_build, tau_build)`` fuction need some explanation. ``k_build`` describes the number of outgoing edges per node in the graph, the larger ``k_build`` the longer the build time and the query. ``tau_build`` influences the stopping criterion during the creation of the graph, the larger the ``tau_build``, the longer the build time. Typically, :math:`0 < tau\_build < 2` is enough to get good results during search. 
It is recommended to experiment with these parameters to get the best possible trade-off between build time and accuracy out of the search. See the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`search parameters <Search_Parameters>` section for more information on parameters and some examples.

Now, the approximate nearest neighbor search can be performed. In this example, a groundtruth is computed via a bruteforce query and the result of the ANN search is evaluated:

.. code:: python

   #run query
   k_query: int = 10
   
   indices, dists = my_ggnn.query(query, k_query=k_query, tau_query=0.9, max_iterations=1000)
   
   
   #run bruteforce query to get a groundtruth and evaluate the results of the query
   gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query)
   evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
   print(evaluator.evaluate_results(indices))


``query(query, k_query, tau_query, max_iterations)`` takes ``query`` (the data to query for), ``k_query`` (the number of neighbors to search), ``tau_query`` and ``max_iterations``. To fine-tune performance for your application you should play around with these parameters. Refer to the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`Search Parameters <Search_Parameters>` section for more information about parameters and some examples. The ``Evaluator`` class holds the necessary information for evluating the results of the query. the function ``evaluate_results(indices)`` compares the results of the query (``indices``) with the results from the bruteforce query (``indices_eval``). 

We can also look at the indices of the *k*-nearest neighbors for the first five queries and their squared euclidean distance:

.. code:: python

   #print the indices of the 10 NN of the first five queries and their squared euclidean distances 
   print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

Usage with Data on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

This  works just like with data on the host, you just have to change the device of your torch tensors to ``device='cuda'`` and potentially add the respective gpu index. Additionally you can tell ggnn to return the result of the *k*-nearest neighbor search on the GPU with ``my_ggnn.set_return_results_on_gpu(True)``.

.. code:: python

   #initialize data
   base = torch.rand((100000, 128), dtype=torch.float32, device='cuda')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cuda')

   #initialize ggnn
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.set_return_results_on_gpu(True)

.. note::
   The data has to be given on the same GPU as the search should be performed on, if your data is sitting on a different GPU you have to move it first.


Usage Multi-GPU
~~~~~~~~~~~~~~~

For multi-gpu mode it is required to use ``set_shard_size(N_shard)``, where ``N_shard`` describes the number of base vectors that should be processed at once. Also the GPU ids have to be provided via ``set_gpus()``, which expects a list of GPU ids.

.. code:: python
   
   #! /usr/bin/python3
   
   import ggnn
   import torch
   
   k_query: int = 10
   
   #initialize data
   base = torch.rand((1000000, 128), dtype=torch.float32, device='cpu')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cpu')
   
   #initialize ggnn and prepare multi gpu
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.set_shard_size(125000)
   my_ggnn.set_gpus([0,1])
   
   #build the graph
   my_ggnn.build(64, 0.9)
   
   #run query
   indices, dists = my_ggnn.query(query, k_query, 0.9, 1000)
   
   print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

.. note::
   The ``Evaluator`` class is only available in single-gpu mode.

Usage of Datasets (e.g. SIFT1M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to query datasets in :file:`.bvecs` or :file:`.fvecs` format, you can use the ``.load('path to file')`` function to load the dataset. If a groundtruth is provided you can pass it to the ``Evaluator``.

.. code:: python

   #! /usr/bin/python3
   
   import ggnn
   
   base = ggnn.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_base.fvecs')
   query = ggnn.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_query.fvecs')
   gt = ggnn.IntDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_groundtruth.ivecs')
   
   k_query: int = 10
   
   evaluator = ggnn.Evaluator(base, query, gt, k_query)
   
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.build(24, 0.5)
   
   indices, dists = my_ggnn.query(query, k_query, 0.64, 400)
   print(evaluator.evaluate_results(indices))


Usage in C++
------------

Standard Usage
~~~~~~~~~~~~~~

You can find all the code from this tutorial and additional example files in the :file:`examples/` folder of the GGNN repository.

Before using ggnn, we need to include ``ggnn/base/ggnn.cuh`` from the ggnn library. The header files from the standard library are only for demonstrtaing purposes and are not required for using the library. Then, some data to search in and some data to search the *k*-nearest neighbors for is needed. Instead of a ``std:array`` you can also use a ``std::vector``:

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

Usage Multi-GPU
~~~~~~~~~~~~~~~

To work on multiple GPUs, we need to pass a ``std::vector<int>`` of GPU ids. Additionally, we need to set ``shard_size``. 
If we use multiple gpus, a gpu deals with one part of the dataset at once and the parts are being swapped out. Therefore, the size of the base dataset has to be evenly divisible by ``shard_size``. The code could look as follows:

.. code:: c++

   //initialize ggnn
   GGNN ggnn;
   
   const size_t total_memory = getTotalSystemMemory();
   // guess the available memory (assume 1/8 used elsewhere, subtract dataset)
   const size_t available_memory = total_memory-total_memory/8-base.size_bytes();
   ggnn.setCPUMemoryLimit(available_memory);
   
   ggnn.setWorkingDirectory(FLAGS_graph_dir);
   ggnn.setBaseReference(base);  
   
   //only necessary in multi-gpu mode
   std::vector<int> gpus = {0,1};
   const uint32_t shard_size = 1000000
   ggnn.setGPUs(gpus);
   ggnn.setShardSize(shard_size);


Usage Datasets (e.g. SIFT1M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can also query for benchmark datasets like `SIFT1M, SIFT1B,...<http://corpus-texmex.irisa.fr/>` in :file:`.bvecs` or :file:`.fvecs` format. We just need to include some extra headers for parsing information from the command line. Additionally ``getTotalSystemMemory()`` helps to manage the memory of our machine properly, especially if we deal with large datasets.

.. code:: c++

   #include <gflags/gflags.h>
   #include <glog/logging.h>
   #include <cstdint>
   #include <cstddef>
   #include <cstdlib>
   
   #include <filesystem>
   
   #include <iostream>
   #include <vector>
   #include <sstream>
   #include <iterator>
   #include <limits>
   #include <string>
   
   #include <ggnn/base/ggnn.cuh>
   #include <ggnn/base/eval.h>
   // only needed for getTotalSystemMemory()
   #include <unistd.h>
   
   using namespace ggnn;
   
   DEFINE_string(base, "", "path to file with base vectors");
   DEFINE_string(query, "", "path to file with query vectors");
   DEFINE_string(gt, "","path to file with groundtruth vectors");
   DEFINE_string(graph_dir, "", "directory to store and load ggnn graph files.");
   DEFINE_double(tau, 0.5, "Parameter tau");
   DEFINE_uint32(refinement_iterations, 2, "Number of refinement iterations");
   DEFINE_uint32(k_build, 24, "Number of neighbors for graph construction");
   DEFINE_uint32(k_query, 10, "Number of neighbors to query for");
   DEFINE_string(measure, "euclidean", "distance measure (euclidean or cosine)");
   DEFINE_uint32(shard_size, 0, "Number of vectors per shard");
   DEFINE_uint32(subset, 0, "Number of base vectors to use");
   DEFINE_string(gpu_ids, "0", "GPU id");
   DEFINE_bool(grid_search, false, "Perform queries for a wide range of parameters.");

   size_t getTotalSystemMemory()
   {
       size_t pages = sysconf(_SC_PHYS_PAGES);
       size_t page_size  = sysconf(_SC_PAGE_SIZE);
       return pages * page_size;
   }

   int main(int argc, char* argv[]) {
     google::InitGoogleLogging(argv[0]);
     google::LogToStderr();
     google::InstallFailureSignalHandler();
   
     gflags::SetUsageMessage(
         "GGNN: Graph-based GPU Nearest Neighbor Search\n"
         "by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. "
         "Lensch\n"
         "(c) 2020 Computer Graphics University of Tuebingen");
     gflags::SetVersionString("1.0.0");
     gflags::ParseCommandLineFlags(&argc, &argv, true);
   
     LOG(INFO) << "Reading files";
     CHECK(std::filesystem::exists(FLAGS_base))
         << "File for base vectors has to exist";
     CHECK(std::filesystem::exists(FLAGS_query))
         << "File for query vectors has to exist";
     CHECK(std::filesystem::exists(FLAGS_gt))
         << "File for groundtruth vectors has to exist";
   
     CHECK_GE(FLAGS_tau, 0) << "Tau has to be bigger or equal 0.";
     CHECK_GE(FLAGS_refinement_iterations, 0)
         << "The number of refinement iterations has to be non-negative.";

Then, we configure the data types we need, read the distance measure and the gpus. For SIFT1B for example, the ``using BaseT = float;`` has to be replaced by ``using BaseT = char;``: 

.. code:: c++

     // data types
     //
     /// data type for addressing points (needs to be able to represent N)
     using KeyT = int32_t;
     /// data type of the dataset (e.g., char, int, float)
     using BaseT = float;
     /// data type of computed distances
     using ValueT = float;
   
     using GGNN = GGNN<KeyT, ValueT, BaseT>;
     using Results = ggnn::Results<KeyT, ValueT>;
     using Evaluator = ggnn::Evaluator<KeyT, ValueT, BaseT>;
   
     /// distance measure (Euclidean or Cosine)
     const DistanceMeasure measure = [](){
       if(FLAGS_measure == "euclidean"){
         return Euclidean;
       }
       else if (FLAGS_measure == "cosine") {
         return Cosine;  
       }
       LOG(FATAL) << "invalid measure: " << FLAGS_measure;
     }();
   
     //vector of gpu ids
     std::istringstream iss(FLAGS_gpu_ids);
     std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                      std::istream_iterator<std::string>());
   
     std::vector<int> gpus;
     for (auto&& r : results) {
       int gpu_id = std::atoi(r.c_str());
       gpus.push_back(gpu_id);
     }

Then, we can load the datasets:

.. code:: c++

   //base & query datasets
   Dataset<BaseT> base = Dataset<BaseT>::load(FLAGS_base, 0, FLAGS_subset ? FLAGS_subset : std::numeric_limits<uint32_t>::max(), true);
   Dataset<BaseT> query = Dataset<BaseT>::load(FLAGS_query, 0, std::numeric_limits<uint32_t>::max(), true);

And can initialize ggnn:

.. code:: c++

   //initialize ggnn
   GGNN ggnn;
   
   const size_t total_memory = getTotalSystemMemory();
   // guess the available memory (assume 1/8 used elsewhere, subtract dataset)
   const size_t available_memory = total_memory-total_memory/8-base.size_bytes();
   ggnn.setCPUMemoryLimit(available_memory);
   
   ggnn.setWorkingDirectory(FLAGS_graph_dir);
   ggnn.setBaseReference(base);

We load the graph if it was built before, else we build and store it:

.. code:: c++
   
   //build the graph
   if (!FLAGS_graph_dir.empty() && std::filesystem::is_regular_file(std::filesystem::path{FLAGS_graph_dir} / "part_0.ggnn")) {
      ggnn.load(FLAGS_k_build);
   }
   else {
    ggnn.build(FLAGS_k_build, static_cast<float>(FLAGS_tau), FLAGS_refinement_iterations, measure);
   
    if (!FLAGS_graph_dir.empty()) {
      ggnn.store();
    }
   }

We obtain the groundtruth:


.. code:: c++
   
   //load or compute groundtruth
   const bool loadGT = std::filesystem::is_regular_file(FLAGS_gt);
   Dataset<KeyT> gt = loadGT ? Dataset<KeyT>::load(FLAGS_gt) : Dataset<KeyT>{};
   
   if (!gt.data()) {
      gt = ggnn.bfQuery(query).ids;
      if (!FLAGS_gt.empty()) {
          LOG(INFO) << "exporting brute-forced ground truth data.";
          gt.store(FLAGS_gt);
      }
   }
   
   Evaluator eval {base, query, gt, FLAGS_k_query, measure};

Finally, we can perform the query:


.. code:: c++
   
   //query
   auto query_function = [&ggnn, &eval, &query, measure](const float tau_query) {
    Results results;
    LOG(INFO) << "--";
    LOG(INFO) << "Query with tau_query " << tau_query;
    // faster for C@1 = 99%
    LOG(INFO) << "fast query (good for C@1)";
    results = ggnn.query(query, FLAGS_k_query, tau_query, 200, measure);
    LOG(INFO) << eval.evaluateResults(results.ids);
    // better for C@10 > 99%
    LOG(INFO) << "regular query (good for C@10)";
    results = ggnn.query(query, FLAGS_k_query, tau_query, 400, measure);
    LOG(INFO) << eval.evaluateResults(results.ids);
    // expensive, can get to 99.99% C@10
    // ggnn.queryLayer<KQuery, 2000, 2048, 256>();
   };
   
   if (FLAGS_grid_search) {
    LOG(INFO) << "--";
    LOG(INFO) << "grid-search:";
    for (int i = 0; i < 70; ++i)
      query_function(static_cast<float>(i) * 0.01f);
    for (int i = 7; i <= 20; ++i)
      query_function(static_cast<float>(i) * 0.1f);
   } else {  // by default, just execute a few queries
    LOG(INFO) << "--";
    LOG(INFO) << "90, 95, 99% R@1, 99% C@10 (using -tau 0.5 "
                 "-refinement_iterations 2):";
    query_function(0.34f);
    query_function(0.41f);
    query_function(0.51f);
    query_function(0.64f);
   }
   
   VLOG(1) << "done!";
   gflags::ShutDownCommandLineFlags();
   return 0;
   }
