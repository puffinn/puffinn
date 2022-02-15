# Microbenchmarks

These microbenchmarks rely on the [`nanobench`](https://nanobench.ankerl.com/) framework, and are implemented in the `bench/bench.cpp` file.
The goal is to inspect the behavior of different components, so to set a baseline for optimization.

All of the following tests have been run on the first 10000 vectors of the [`glove.twitter.25B`](https://nlp.stanford.edu/data/glove.twitter.27B.zip) dataset, with 100 dimensions.
The times reported are obtained on a Macbook Pro (Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz), with code compiled with Clang 10.0.1.

## Index construction

We measure the time to insert the data into an emtpy index, and the time to rebuild said index with the newly added data.
Therefore the time to build the prefix maps is the difference between the two.

|               ms/op |                op/s |    err% |     total | Index building
|--------------------:|--------------------:|--------:|----------:|:---------------
|               13.08 |               76.47 |    2.1% |      0.15 | `index_insert_data`
|            7,157.27 |                0.14 |    6.5% |     80.64 | :wavy_dash: `index_rebuild` (Unstable with ~1.0 iters. Increase `minEpochIterations` to e.g. 10)

As you can see from the table above, the time to insert data is negligible compared to the 7 seconds it takes to build the prefix maps.

## Querying the index

Here we test the time it takes to query the index, using three different query vectors and asking for the 10 nearest neighbors with 0.9 guaranteed recall.
Times are in nanoseconds in this table, hence we have that all queries take around one millisecond.

|               ns/op |                op/s |    err% |     total | Index query
|--------------------:|--------------------:|--------:|----------:|:------------
|        1,101,686.88 |              907.70 |    0.4% |      1.33 | `index_query (query 0)`
|        1,139,939.24 |              877.24 |    2.2% |      1.38 | `index_query (query 100)`
|        1,076,122.38 |              929.26 |    0.9% |      1.30 | `index_query (query 1000)`

## Hashing

The code for this benchmark is [here](https://github.com/Cecca/puffinn/blob/3142c5d2c0e101bcfce119cd33d98e7250ab3aa1/bench/bench.cpp#L106-L123).
During index construction the hash function is hidden behind a unique pointer, and the vector to be hashed as well.
This indirection may be expensive, therefore we try also a direct static implementation. 
In all benchmarks we compute 24-bits hashes, with the exeption of the `single` calls that generate 8 bits for cross polytope and 1 for simhash.
Therefore, for cross polytope we would expect times to be three times higher than a single invocation, for simhash 24 times higher.
Again, the times are in nanoseconds.

|               ns/op |                op/s |    err% |     total | Hashing
|--------------------:|--------------------:|--------:|----------:|:--------
|              995.79 |        1,004,225.20 |    1.2% |      0.01 | `Cross polytope (single)`
|            3,073.02 |          325,412.89 |    1.4% |      0.04 | `Cross polytope (indirection)`
|            3,046.00 |          328,299.19 |    0.1% |      0.04 | `Cross polytope (static)`
|              321.03 |        3,114,996.82 |    0.0% |      0.00 | `FHT Cross polytope (single)`
|            1,044.19 |          957,681.46 |    1.0% |      0.01 | `FHT cross polytope (indirection)`
|              934.88 |        1,069,653.37 |    0.1% |      0.01 | `FHT cross polytope (static)`
|                5.03 |      198,865,506.85 |    0.2% |      0.00 | `SimHash (single)`
|              332.74 |        3,005,358.55 |    3.1% |      0.00 | `SimHash (indirection)`
|              260.93 |        3,832,376.04 |    4.8% |      0.00 | `SimHash (static)`
