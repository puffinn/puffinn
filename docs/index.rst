.. PUFFINN documentation master file, created by
   sphinx-quickstart on Sun Jun 30 22:07:33 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PUFFINN 
=======

**Parameterless and Universal Fast FInding of Nearest Neighbors**

PUFFINN is a library which uses Locality Sensitive Hashing (LSH)
to find close neighbors in high-dimensional space.
The specific approach used is described below and has a number of desirable properties,
namely that it is easy to use and provides guarantees for the expected accuracy of the result.
It is also very performant.
Knowledge of LSH is not necessary to use the library
when using the recommended default parameters in ``Index``.

Locality Sensitive Hashing
--------------------------
A more thorough introduction to LSH should be found elsewhere,
but a brief overview is presented here.

Locality Sensitive Hash functions are hash functions with the property that similar points
have a higher probability of having a hash collision than distant points.
These hash functions are grouped into families, from which multiple functions are drawn randomly.
By concatenating multiple such functions, the collision probability of distant points
becomes very low, while the collision probability of similar points remains reasonably high.

To use these functions to search for nearest neighbors, it is first necessary to construct an index
which contains hashes of every point for multiple concatenated LSH functions.
The nearest neighbors of a query point can then be found by hashing the query
using the same hash functions.
Any point with a colliding hash is then considered a candidate for
being among the nearest neighbors.
The most similar points among the candidates are then found by computing the actual similarity.
In this way, most of the dataset does not need to be considered.

Algorithm Description
---------------------
PUFFINN uses an adaptive query mechanism which adjusts the length of the concatenated hashes
depending on the similarity of the nearest neighbors and the index size.
If the index is small or if the neighbors are distant, the collision probability needs to be
increased, which is done by reducing the length of the hashes.
This ensures that the target recall is achieved regardless of the difficulty of the query.

PUFFINN also uses a filtering step to further reduce the number of candidates.
This is done by computing a number of sketches for each point using 1-bit LSH functions.
Points are then only considered if the hamming similarity of a randomly selected pair of sketches
is above a set treshold, which depend on the similarity.
This significantly reduces the number of candidates, but reduces the recall slightly.

It is also possible to draw hash functions from different sources.
By default, they are sampled independently, but it is also possible to use a precalculated
set of functions and concatenate them in various ways.
This reduces the number of necessary hash computations but yields hashes of lower quality.
It is sometimes more efficient to use one of these alternative hash sources.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

C++ Documentation
=================

.. doxygenclass:: puffinn::Index
   :members:
.. doxygenstruct:: puffinn::CosineSimilarity
   :members: Format, DefaultHash, DefaultSketch
   :undoc-members:
.. doxygenstruct:: puffinn::JaccardSimilarity
   :members: Format, DefaultHash, DefaultSketch
   :undoc-members:
.. doxygenstruct:: puffinn::UnitVectorFormat
.. doxygenstruct:: puffinn::SetFormat
.. doxygenclass:: puffinn::SimHash
   :members: Args, Format
   :undoc-members:
.. doxygenstruct:: puffinn::SimHashArgs
.. doxygenclass:: puffinn::CrossPolytopeHash
   :members: Args, Format
   :undoc-members:
.. doxygenstruct:: puffinn::CrossPolytopeArgs
   :members:
.. doxygenclass:: puffinn::FHTCrossPolytopeHash
   :members: Args, Format
   :undoc-members:
.. doxygenstruct:: puffinn::FHTCrossPolytopeArgs
   :members:
.. doxygenclass:: puffinn::MinHash
   :members: Args, Format
   :undoc-members:
.. doxygenstruct:: puffinn::MinHashArgs
.. doxygenclass:: puffinn::MinHash1Bit
   :members: Args, Format
   :undoc-members:
.. doxygenstruct:: puffinn::HashSourceArgs
.. doxygenstruct:: puffinn::IndependentHashArgs
   :members: args
.. doxygenstruct:: puffinn::HashPoolArgs
   :members: args, pool_size, HashPoolArgs
   :undoc-members:
.. doxygenstruct:: puffinn::TensoredHashArgs
   :members: args
.. doxygenenum:: puffinn::FilterType

Python Documentation
====================
.. py:module:: puffinn

.. py:class:: Index(similarity_measure, dimensions, memory_limit, kwargs)

   An index constructed over a dataset which supports approximate near-neighbor queries for a specific similarity measure.
   It can be serialized using pickle.

   :param str metric: The name of the metric used to measure the similarity of two points. Currently ``"angular"`` and ``"jaccard"`` are supported, which respectively map to ``CosineSimilarity`` and ``JaccardSimilarity`` in the C++ API.
   :param integer dimensions: The required number of dimensions of the input. When using the ``"angular"`` metric, all input vectors must have this length. When using the ``"jaccard"`` metric, all tokens in the input sets must be integers between 0, inclusive, and dimensions, exclusive. 
   :param integer memory_limit: The number of bytes of memory that the index is permitted to use. Using more memory almost always means that queries are more efficient. 
   :param kwargs: Additional arguments used to setup hash functions. None of these are necessary. The hash family, hash source and their arguments are given by specifying ``"hash_function"``, ``"hash_args"``, ``"hash_source"`` and ``"source_args"`` respectively.
   :param kwargs.hash_function: The hash function can be either ``"simhash"``, ``"crosspolytope"``, ``"fht_crosspolytope"``, ``"minhash"`` or ``"1bit_minhash"``, depending on the metric. See the C++ documentation on the corresponding types for details.
   :param kwargs.hash_args: Arguments for the used hash function. The supported arguments when using "crosspolytope" are "estimation_repetitions" and "estimation_eps". Using "fht_crosspolytope", "num_rotations" can also be specified. The other hash functions do not take any arguments. See the C++ documentation on the hash functions for details.
   :param kwargs.hash_source: The supported hash sources are ``"independent"``, ``"pool"`` and ``"tensor"``. See the C++ documentation on ``HashSourceArgs`` for details.
   :param kwargs.source_args: Arguments for the hash source. Most hash sources do not take arguments. If ``"pool"`` is selected, the size of the pool can be specified as the ``"pool_size"``.

   .. py:method:: insert(value)

   Insert a value into the index.

   Before the value can be found using the search method, :py:meth:`rebuild` must be called.

   :param list[integer] value: The value to insert.

   .. py:method:: get(idx)

   Retrieve a value that has been inserted into the index.

   The value is converted back from the internal storage format, which means that it is unlikely to be equal to the inserted value due to normalization, rounding and other adjustments.

   :param integer idx: The value to retrieve by insertion order.

   .. py:method:: rebuild()

    Rebuild the index using the currently inserted points.

    This is done in parallel.

   .. py:method:: search(query, k, recall, filter_type = "default")

   Search for the approximate k nearest neighbors to a query.

   :param list[integer] query: The query value.
   :param integer k: The number of neighbors to search for.
   :param float recall: The expected recall of the result. Each of the nearest neighbors has at least this probability of being found in the first phase of the algorithm. However if sketching is used, the probability of the neighbor being returned might be slightly lower. This is given as a number between 0 and 1. 
   :param string filter_type: The approach used to filter candidates. Unless the expected recall needs to be strictly above the recall parameter, the default should be used. The suppported types are "default", "none" and "simple". See ``FilterType`` for more information. 
