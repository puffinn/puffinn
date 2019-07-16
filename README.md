[![Build Status](https://travis-ci.com/puffinn/puffinn.svg?branch=master)](https://travis-ci.com/puffinn/puffinn)

# PUFFINN - Parameterless and Universal Fast FInding of Nearest Neighbors
PUFFINN is an easily configurable library for finding the approximate nearest neighbors of arbitrary points.
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
To use the library, use the `insert`, `rebuild` and `search` methods on `puffinn::Index` as shown in the below example. 
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
}
```

## Python
To build the library locally using setuptools, run `python3 setup.py build`. 

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
```

# Authors

PUFFINN is mainly developed by Michael Vesterli. It grew out of a research project with Martin Aumüller, Tobias Christiani, and Rasmus Pagh. If you want to cite PUFFINN in your publication, please use the following reference.

> PUFFINN: Parameterless and Universal Fast FInding of Nearest Neighbors, M. Aumüller, T. Christiani, R. Pagh, and M. Vesterli. ESA 2019.

An extended version of the paper is available at https://arxiv.org/abs/1906.12211.

