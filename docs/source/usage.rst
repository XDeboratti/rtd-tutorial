Usage
=====

This section explains how to use the ggnn module and the ggnn library. The ggnn library is written for easy use in Python, but of course using it from C++ is also explained here.


Usage in Python
---------------

The code from this tutorial and additional examples can be found in the :file:`ggnn/examples/python/ggnn_pytorch.py` file of the GGNN repository.

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
   my_ggnn.build(k_build=64, tau_build=0.9, measure=ggnn.DistanceMeasure.Euclidean)

The parameters of the ``build(k_build, tau_build, measure)`` fuction need some explanation. ``k_build >= 2`` describes the number of outgoing edges per node in the graph, the larger ``k_build`` the longer the build time and the query. ``tau_build`` influences the stopping criterion during the creation of the graph, the larger the ``tau_build``, the longer the build time. Typically, :math:`0 < tau\_build < 2` is enough to get good results during search. 
It is recommended to experiment with these parameters to get the best possible trade-off between build time and accuracy out of the search. See the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`search parameters <Search_Parameters>` section for more information on parameters and some examples.
``measure`` is the distance measure to compare the distances of the vectors. The ggnn module supports cosine and euclidean (L2) distance, euclidean distance is the default, so passing this parameter is optional.

Now, the approximate nearest neighbor search can be performed. In this example, a groundtruth is computed via a bruteforce query and the result of the ANN search is evaluated:

.. code:: python

   #run query
   k_query: int = 10
   
   indices, dists = my_ggnn.query(query, k_query=k_query, tau_query=0.9, max_iterations=1000, measure=ggnn.DistanceMeasure.Euclidean)
   
   
   #run bruteforce query to get a groundtruth and evaluate the results of the query
   gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query, measure=ggnn.DistanceMeasure.Euclidean)
   evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
   print(evaluator.evaluate_results(indices))

The parameters of ``query(query, k_query, tau_query, max_iterations, measure)`` are:

- ``query`` are all the vectors, to search the *k*-NN for.
- ``k_query`` tells the search algorithm how many neighbors it should return per query vector. Generally, the higher ``k_query``, the longer the search. The ggnn module supports up to 6000 neighbors, but it is recommended to search only for 1-1000 neighbors.
- ``tau_query`` and ``max_iterations`` determine the stopping criterion. For both parameters it holds that the larger the parameter, the longer the search. Typically, :math:`0 < tau\_query < 2` and :math:`0 < max\_iterations < 2000` is enough to get good results during search.
- ``measure`` is the distance measure that is used to compute the distances between vectors. ``Euclidean`` is the default, so this parameter is optional. To set cosine similarity you can pass ``measure=ggnn.DistanceMeasure.Cosine`` as parameter. 

For computing a groundtruth, we need  to pass ``k_gt`` which should be the same as ``k_query`` if we want to compare properly.

.. caution::

   The distance measure for building, querying and computing the groundtruth should be the same.

After evaluating the example program prints the indices of the *k*-nearest neighbors for the first five queries and their squared euclidean distances:

.. code:: python

   #print the indices of the 10 NN of the first five queries and their squared euclidean distances 
   print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

Usage with Data on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

This works just like with data on the host, but the device of the torch tensors must be set to ``device='cuda'`` and possibly the respective gpu index must be added. Additionally, ggnn can return the result of the *k*-nearest neighbor search on the GPU with ``my_ggnn.set_return_results_on_gpu(True)``. If not set, the results will be on the host.

.. code:: python

   #create data
   base = torch.rand((100000, 128), dtype=torch.float32, device='cuda')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cuda')

   #initialize ggnn
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.set_return_results_on_gpu(True)

.. note::
   The data has to be given on the same GPU as the search should be performed on, if the data is sitting on a different GPU it needs to be moved first.


Usage Multi-GPU
~~~~~~~~~~~~~~~

For multi-gpu mode it is required to use ``set_shard_size(n_shard)``, where ``n_shard`` describes the number of base vectors that should be processed at once. Also the GPU ids have to be provided via ``set_gpus(gpu_ids)``, which expects a list of GPU ids. 

.. code:: python
   
   #! /usr/bin/python3
   
   import ggnn
   import torch
   
   #create data
   base = torch.rand((1000000, 128), dtype=torch.float32, device='cpu')
   query = torch.rand((10000, 128), dtype=torch.float32, device='cpu')
   
   #initialize ggnn and prepare multi gpu
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.set_shard_size(n_shard=125000)
   my_ggnn.set_gpus(gpu_ids=[0,1])
   
   #build the graph
   my_ggnn.build(k_build=64, tau_build=0.9)
   
   #run query
   k_query: int = 10
   
   indices, dists = my_ggnn.query(query, k_query=k_query, tau_query=0.9, max_iterations=1000)
   
   print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

.. caution::
   When using multiple GPUs for the search, data has to be copied through the cpu before it can be spreaded on multiple GPUs.
   Also returning the results is only possible on the host side when using multiple GPUs (for now).

.. note::
   The ``Evaluator`` class is only available in single-gpu mode. 

Usage of Datasets (e.g. SIFT1M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the data is provided in :file:`.bvecs` or :file:`.fvecs` format, as for example the SIFT1M dataset, the dataset can be loaded using the ``.load('path to file')`` function. Besides a ``FloatDataset``, the ggnn module can also load a base and query as ``UCharDataset`` (unsigned char). If a groundtruth is provided it can be passed to the ``Evaluator`` directly.

.. code:: python

   #! /usr/bin/python3
   
   import ggnn
   
   base = ggnn.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_base.fvecs')
   query = ggnn.FloatDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_query.fvecs')
   gt = ggnn.IntDataset.load('/graphics/scratch/datasets/ANN_datasets/SIFT1M/sift/sift_groundtruth.ivecs')
   
   k_query: int = 10
   
   evaluator = ggnn.Evaluator(base, query, gt=gt, k_query=k_query)
   
   my_ggnn = ggnn.GGNN()
   my_ggnn.set_base(base)
   my_ggnn.build(k_build=24, tau_build=0.5)
   
   indices, dists = my_ggnn.query(query, k_query, tau_query=0.64, max_iterations=400)
   print(evaluator.evaluate_results(indices))


Usage in C++
------------

You can find all the code from this tutorial and additional example files in the :file:`examples/cpp-and-cuda/` folder of the GGNN repository.

Standard Usage
~~~~~~~~~~~~~~

Before using ggnn, the ``ggnn/base/ggnn.cuh`` header has to be included from the ggnn library. The header files from the standard library are only for demonstrtaing purposes and are not required for using the library. Then, some data to search in and some data to search the *k*-nearest neighbors for is needed. Instead of a ``std:array`` you can also use a ``std::vector``:

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

Then,  a ggnn instance and the datasets can be initialized:

.. code:: c++

       // data types
       //
       /// data type for addressing points
       using KeyT = int32_t;
       /// data type of the dataset (uint8_t, float)
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

The parameters of the ``build(const uint32_t KBuild, const float tau_build, const uint32_t refinement_iterations, const DistanceMeasure measure);`` fuction need some explanation. ``KBuild >= 2`` describes the number of outgoing edges per node in the graph, the larger ``KBuild`` the longer the build time and the query. ``tau_build`` influences the stopping criterion during the creation of the graph, the larger the ``tau_build``, the longer the build time. Typically, :math:`0 < tau\_build < 2` is enough to get good results during search. 
It is recommended to experiment with these parameters to get the best possible trade-off between build time and accuracy out of the search. See the paper `GGNN: Graph-based GPU Nearest Neighbor Search <https://arxiv.org/abs/1912.01059>`_ and the :ref:`search parameters <Search_Parameters>` section for more information on parameters and some examples.
``refinement_iterations`` is the number of times the refinement algorithm is executed. It was shown empirically that 2 refinement iterations are sufficient to obtain a well connected graph.
``DistanceMeasure`` is the distance measure to compare the distances of the vectors. The ggnn module supports cosine and euclidean (L2) distance, euclidean distance is the default, so passing this parameter is optional. Cosine distance can be passed as parameter ``Cosine`` or by declaring a variable of type ``ggnn::DistanceMeasure`` before.

.. code:: c++

       //call query and store indices & squared distances
       const uint32_t KQuery = 10;
       const auto [indices, dists] = ggnn.query(query, KQuery, 0.5);

The parameters of ``query(const Dataset<BaseT>& query, const uint32_t KQuery, const float tau_query, const uint32_t max_iterations, const DistanceMeasure measure)`` are:

- ``query`` are all the vectors, to search the *k*-NN for.
- ``KQuery`` tells the search algorithm how many neighbors it should return per query vector. Generally, the higher ``KQuery``, the longer the search. The ggnn module supports up to 6000 neighbors, but it is recommended to search only for 1-1000 neighbors.
- ``tau_query`` and ``max_iterations`` determine the stopping criterion. For both parameters it holds that the larger the parameter, the longer the search. Typically, :math:`0 < tau\_query < 2` and :math:`0 < max\_iterations < 2000` is enough to get good results during search. For ``max_iterations`` the default is set to 400.
- ``measure`` is the distance measure that is used to compute the distances between vectors. ``Euclidean`` is the default, so this parameter is optional. To set cosine similarity you can pass ``Cosine`` as parameter. 

The example program prints the indices and squared euclidean distances of the 10 nearest neighbors of the first query:

.. code:: c++

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

To work on multiple GPUs, the method ggnn.setGPUs(const std::span<const int>& gpu_ids) has to be used to tell the instance of the ggnn class which GPUs to use. Additionally, ``ggnn.setShardSize(const uint32_t N_shard)`` needs to tell the ggnn instance how large each shard should be. A gpu deals with one part of the dataset (shard) at once and the parts are being swapped out. Therefore, the size of the base dataset has to be evenly divisible by ``shard_size``. The code could look as follows:

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

   //buid the kNN graph
   ggnn.build(24, 0.5);


Usage Datasets (e.g. SIFT1M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library also provides functionality to query for benchmark datasets such as `SIFT1M, SIFT1B,...<http://corpus-texmex.irisa.fr/>` in :file:`.bvecs` or :file:`.fvecs` format. Example files for using of the SIFT1M and SIFT1B datasets can be found in the :file:`examples/cpp-and-cuda/` folder. The files can also be used for other datasets, but the parameters have to be adjusted according to the :ref:`search parameters <Search_Parameters>` section. The program can be run as follows:

.. code::

   ./build/sift1m --base ./path-to-dataset/sift_base.fvecs --query /path-to-dataset/sift_query.fvecs --gt /path-to-dataset/sift_groundtruth.ivecs --graph_dir ./ --tau 0.5 --refinement_iterations 2 --k_build 24 --k_query 10 --measure euclidean --shard_size 0 --subset 0 --gpu_ids 0 --grid_search false

The ``--graph_dir``, ``--tau``, ``--refinement_iterations``, ``--k_build``, ``--k_query``, ``--measure``, ``--subset`` and ``--grid_search`` flags are optional. The ``--grid_search`` flag is useful for finding the configuration that leads to 99% precision, measured in recall or consensus. The ``--subset`` flag describes the total number of base vectors and is useful if only a subset of the base vectors is to be searched. The ``--shard_size`` and ``--gpu_ids`` flags are optional and are only needed for multi-gpu execution. ``--shard_size`` describes the number of base vectors to process at once. The total number of base vectors must be evenly divisible by the shard size. ``--gpu_ids`` expects a comma-separated list of GPU indices e.g. ``--gpu_ids 0,1,2`` of the GPUs on which the query should be executed.

In the following, the :file:`sift1m` program is explained in more detail, the :file:`sift1b` program works analogously.

Some extra headers are included for parsing information from the command line. Additionally ``getTotalSystemMemory()`` helps to manage the memory of our machine properly.

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

Then, the needed data types are configured for convenience, and the distance measure and the gpu_ids are read. For SIFT1B for example, the ``using BaseT = float;`` has to be replaced by ``using BaseT = uint8_t;`` or ``using BaseT = unsigned char;``: 

.. code:: c++

     // data types
     //
     /// data type for addressing points (needs to be able to represent N)
     using KeyT = int32_t;
     /// data type of the dataset (e.g., char, float)
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

Then, the datasets are loaded:

.. code:: c++

   //base & query datasets
   Dataset<BaseT> base = Dataset<BaseT>::load(FLAGS_base, 0, FLAGS_subset ? FLAGS_subset : std::numeric_limits<uint32_t>::max(), true);
   Dataset<BaseT> query = Dataset<BaseT>::load(FLAGS_query, 0, std::numeric_limits<uint32_t>::max(), true);

And a ggnn instance is created:

.. code:: c++

   //initialize ggnn
   GGNN ggnn;
   
   const size_t total_memory = getTotalSystemMemory();
   // guess the available memory (assume 1/8 used elsewhere, subtract dataset)
   const size_t available_memory = total_memory-total_memory/8-base.size_bytes();
   ggnn.setCPUMemoryLimit(available_memory);
   
   ggnn.setWorkingDirectory(FLAGS_graph_dir);
   ggnn.setBaseReference(base);

The graph is loaded if it was built before, else it build and potentially stored:

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

Fro evaluation of the algorithm, a groundtruth is needed, so it is loaded or if there is not groundtruth file the groundtruth is computed:


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

Finally, the query is performed. If the ``--grid_search`` flag is set, the ``tau_query`` parameter is slowly increased. If not, just a few ``tau_queries`` are applied:

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
