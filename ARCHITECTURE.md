# `PUFFINN`'s architecture

The purpose of this document is to outline the organization of `PUFFINN`'s codebase.

The goal of `PUFFINN` is to build indices to support efficient similarity
search queries under a variety of distance functions. The main technique being
used is Locality Sensitive Hashing (LSH).

## Data representation

`PUFFINN` works with dense vectors and sparse sets.
All headers related to data representation are in `include/puffinn/format`.

- General dense vectors are implemented in `real_vector.hpp`.
- Dense vectors of unit norm are implemented in `unit_vector.hpp`. Since by
construction they have unit norm then they are represented using 16-bits fixed
point numbers rather than floating point numbers. This makes for both less
space and faster operation.
- Sets are implemented in `set.hpp` as sorted vector of token identifiers,
ranging from 0 to the size of the universe set.

## Distance functions

`PUFFINN` supports the following distance functions, all implemented in the
`include/puffinn/similarity_measure` directory. Importantly,

- Cosine similarity (`cosine.hpp`) assumes to be working with unit vectors
(i.e. `UnitVectorFormat`), the implementation **does not work** with
`RealVectorFormat`.
- Jaccard similarity (`jaccard.hpp`) assumes to be working with `SetFormat`.

Both implementation work directly with the raw data wrapped by
`UnitVectorFormat` and `SetFormat`.

## Hash functions

The `include/puffinn/hash` directory contains the implementations of LSH functions:

- `simhash.hpp` and `cross_polytope.hpp` for cosine similarity
- `minhash.hpp` for Jaccard similarity

All of the above implementations define a class encapsulating each hash
function's state (e.g. `SimHashFunction`) and defining an `operator()` method
that takes a single data item (i.e. a vector or a set) and produces an
`LshDatatype` return value. Note that a _single_ invocation of `operator()`
produces a _single_ hash value (e.g. a single bit in `SimHashFunction`).
These hash values are meant to be combined by Hash Sources (see next section).

The `*Function` objects are built by means of factories providing a method
`sample()` (e.g. the class `SimHash`).
These factories are configured upon creation using instances of `*Args` classes (e.g. `SimHashArgs`).

## Hash sources

LSH hash values are rarely useful in isolation. In `PUFFINN` we require `K`
concatenations and `L` repetitions. The purpose of a hash source is to
construct these `L` repetitions of `K` concatenated hash values.

All hash sources extend a common abstract class (`HashSource`, defined in
`hash_source.hpp`) 
Instances of Hash sources are built by means of objects extending the
`HashSourceArgs` abstract class, which act as factory objects.

`HashSource` exposes the following methods:

- `hash_repetitions` that accepts a data item to hash (according to the hash
functions described in the previous section) and a vector of 64-bits integers
to hold the output. Each 64 bit integer is the result of concatenating several
hash values, and the output vector holds as many concatenated hashes as there
are repetitions. Note that the 64 bits are not always all used. This method is
in fact called both when computing hash values (where 24 bits are used) and
when computing sketches (where 64 bits are used).
- `collision_probability`, `failure_probability`,
`concatenated_collision_probability` are used when implementing stopping
conditions.

## Sketches

To improve the query efficiency, the `include/puffinn/filterer.hpp` header
implements bitwise sketches. In particular, the `Filterer` class maintains, for
each data item in the dataset, a 64 bit sketches built using one of the hash
sources mentioned previously. The `HashSourceArgs` instance is passed to the
`Filterer` constructor. By default it is `IndependentHashSource` with a hash
function depending on the similarity measure being used. In particular, the
`CosineSimilarity` and `JaccardSimilarity` classes both define `DefaultHash`
and `DefaultSketch` to be the default LSH functions to be used for hashing and
sketching, respectively.

A query vector is skecthed using the `sketch` method that will populate the
`QuerySketches` output parameter, which in turn allows to check whether the
sketches match with an arbitrary sketched vector in the `Filterer` using the
`QuerySketches::passes_filter` method.

## Index

The proper index structure is implemented in `include/puffinn/collection.hpp`,
which contains both the methods to add vectors to the index (`insert` and
`update`) and the query methods (`search_*`).

The index itself maintains a collection of LSH maps, which are implemented in
`prefixmap.hpp`. There is a `PrefixMap` for each repetition. The `PrefixMap`
data structure maintains a vector of indices into the original data and a
vector of the corresponding hash values. Both vectors are sorted in increasing
order of hash value. The `PrefixMap` is then able to provide ranges of indices
whose corresponding hash values share a common prefix of a given length. These
are the indices of the vectors that query algorithms will the compare.

The `maxbuffer.hpp` and `maxpairbuffer.hpp` headers implement priority queues
that are used in the query algorithms.

