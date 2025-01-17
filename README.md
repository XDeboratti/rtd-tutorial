# The ggnn Library
The ggnn library performs nearest-neighbor computations on CUDA-capable GPUs. It supports billion-scale datasets and can execute on multiple GPUs through sharding. It is written in C++ and has Python 3 wrappers, so it can be used from both languages. When using just a single GPU, data can be exchanged directly with other code without copying through CPU memory (e.g., torch tensors). 
The library is based on the method proposed in the paper [GGNN: Graph-based GPU Nearest Neighbor Search](https://arxiv.org/abs/1912.01059) by Fabian Groh, Lukas Ruppert, Patrick Wieschollek, and Hendrik P.A. Lensch. 

<!-- #ToDo: Insert link to docu 
For more detailed information see our [documentation]().-->

<!-- #ToDo: Sollen Leute uns kontaktieren wenn sie Probleme haben? Issues? e-mail?-->

## Installation

This is a short guide on how to install the ggnn python module:

```bash
git clone https://github.com/cgtuebingen/ggnn.git
cd ggnn
pip install .
```

<!--#ToDo: Insert link to Installation
For installation in C++, please see the [documentation]().-->

## Example Usage

One example of how to use the ggnn module is presented here:

```python
#! /usr/bin/python3

import ggnn
import torch

#get detailed logs
ggnn.set_log_level(4)


#create data
base = torch.rand((100000, 128), dtype=torch.float32, device='cpu')
query = torch.rand((10000, 128), dtype=torch.float32, device='cpu')


#initialize ggnn
my_ggnn = ggnn.GGNN()
my_ggnn.set_base(base)

#build the graph
my_ggnn.build(k_build=64, tau_build=0.9, measure=ggnn.DistanceMeasure.Euclidean)


#run query
k_query: int = 10

indices, dists = my_ggnn.query(query, k_query=k_query, tau_query=0.9, max_iterations=1000, measure=ggnn.DistanceMeasure.Euclidean)


#run bruteforce query to get a groundtruth and evaluate the results of the query
gt_indices, gt_dists = my_ggnn.bf_query(query, k_gt=k_query, measure=ggnn.DistanceMeasure.Euclidean)
evaluator = ggnn.Evaluator(base, query, gt_indices, k_query=k_query)
print(evaluator.evaluate_results(indices))

#print the indices of the 10 NN of the first five queries and their squared euclidean distances 
print('indices:', indices[:5], '\n squared dists:',  dists[:5], '\n')

```

<!--#ToDo: Insert link to Usage
For more examples in Python and in C++ see the [examples]() folder. For more information about the parameters, on how to deal with data that is already on a GPU and on how to utilize multiple GPUs, check out the [documentation](). We also provide scripts that load typical benchmark datasets.-->


## Capabilities and Limitations

The ggnn library supports...

- arbitrarily large datasets, the only limit is your hardware.
- data with up to 4096 dimensions
- building graphs with up to 512 edges per node
- searching for up to 6000 nearest neighbors
- two distance measures: cosine and euclidean (L2) distance

---
We hope this library makes your life easier and helps you to solve your problem! 

Happy programming,
@LukasRuppert and [Deborah Kornwolf](https://github.com/XDeboratti)
