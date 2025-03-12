# PUFFINN - Parameterless and Universal Fast FInding of Nearest Neighbors
PUFFINN is an easily configurable library for finding the approximate nearest neighbors of arbitrary points.
It also supports the identification of the closest pairs in the dataset.
The only necessary parameters are the allowed space usage and the recall.
Each near neighbor is guaranteed to be found with the probability given by the recall, regardless of the difficulty of the query. 

Under the hood PUFFINN uses Locality Sensitive Hashing with an adaptive query mechanism.
This means that the algorithm works for any similarity measure where a Locality Sensitive Hash family exists.
Currently Cosine similarity is supported using SimHash or cross-polytope LSH and Jaccard similarity is supported using MinHash.

# Usage
PUFFINN is implemented in C++ with Python bindings available. All features are available in both languages. 
To get started quickly, see the below examples, as well as those in the /examples directory.
More details are available in the [documentation](https://puffinn.readthedocs.io/en/latest/).

## C++
PUFFINN is a header-only library. In most cases, including `puffinn.hpp` is sufficient.
To use the library, use the `insert`, `rebuild`  and `search` methods on `puffinn::Index` as shown in the below example. 
Note that points inserted after the last call to `rebuild` cannot be found.

```cpp
#include "puffinn.hpp"

int main() {
    std::vector<std::vector<float>> dataset = ...;
    int dimensions = ...;
    
    // Construct the index using the cosine similarity measure,
    // the default hash functions and 4 GB of memory.
    puffinn::LSHTable<puffinn::CosineSimilarity> index(dimensions, 4*1024*1024*1024);
    for (auto& v : dataset) { index.insert(v); }
    index.rebuild();
    
    std::vector<float> query = ...;
    
    // Find the approximate 10 nearest neighbors.
    // Each of the true 10 nearest neighbors has at least an 80% chance of being found.
    std::vector<uint32_t> result = index.search(query, 10, 0.8); 
    
    // Find the approximate 10 closest pairs in the dataset.
    // Each of the true 10 closest pairs has at least an 80% chance of being found. 
    std::vector<std::pair<uint32_t, uint32_t>> result = index.closest_pairs(10, 0.8); 
}
```

## Python
To build the library locally using setuptools, run `python3 setup.py build`.
You will first need to install `pybind11` using `python3 -m pip install pybind11`, potentially `python3 -m pip install pybind11[global]`.

The API of the Python wrapper does not differ significantly from C++ API, except that arguments are passed slightly differently. The Python equivalent to the above example is shown below.
See the [documentation](https://puffinn.readthedocs.io/en/latest/) for more details.

```python
import puffinn

dataset = ...
dimensions = ...

# Construct the index using the cosine similarity measure,
# the default hash functions and 4 GB of memory.
index = puffinn.Index('angular', dimensions, 4*1024**3)
for v in dataset:
    index.insert(v)
index.rebuild()

query = ...
    
# Find the approximate 10 nearest neighbors.
# Each of the true 10 nearest neighbors has at least an 80% chance of being found.
result = index.search(query, 10, 0.8) 

# Find the approximate 10 closest pairs in the dataset.
# Each of the true 10 closest pairs has at least an 80% chance of being found.
closest_pairs = index.closest_pairs(k, 0.8)
```

# Choice of Space Usage

PUFFINN will use as much memory as it is allowed to, which generally gives the best query performance for large datasets (n > 400000). However for smaller datasets using less memory can have better performance.
During queries, there is a small query overhead per internal hash table used. For smaller datasets, more tables can be used which increases the overhead.
Another factor to consider is that building the index takes considerable time and is independent of the size of the dataset. This wait might not be worth the small potential speedup of a larger index.

For small datasets, we recommend setting the space usage to at most n^2/60 bytes, where n is the size of the dataset.

# Benchmark

PUFFINN provides fast query times with considerable space usage. It's reliable (see bottom right plot) and doesn't require parameter tuning. 
![Benchmark](https://user-images.githubusercontent.com/6311646/61288829-40903080-a7c8-11e9-9eb0-effc6beb808e.png)

The following benchmark summarizes running times for finding the (globally) $k$-closest pairs in the dataset. 

![Closest Pairs Benchmark](https://github.com/Cecca/puffinn/assets/6311646/b9d96135-0d55-4c01-b00b-60d702312fc3>)

# Authors

PUFFINN is mainly developed by Michael Vesterli. It grew out of a research project with Martin Aumüller, Tobias Christiani, and Rasmus Pagh. If you want to cite PUFFINN in your publication, please use the following reference.

> PUFFINN: Parameterless and Universal Fast FInding of Nearest Neighbors, M. Aumüller, T. Christiani, R. Pagh, and M. Vesterli. ESA 2019.

An extended version of the paper is available at https://arxiv.org/abs/1906.12211.

The closest pair functionality was developed by Martin Aumüller and Matteo Ceccarello. Details of the method are available in the following publication

> Solving $k$-Closest Pairs in High-Dimensional Data, M. Aumüller, M. Ceccarello, SISAP 2023. [Link](https://link.springer.com/chapter/10.1007/978-3-031-46994-7_17)

The experimental setup to reproduce the results from the paper is available in the following repository: <https://github.com/Cecca/puffinn>

