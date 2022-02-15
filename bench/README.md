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

The code for this benchmark is [here](https://github.com/Cecca/puffinn/blob/b7b76f765faf6ad896de67b15b5a41efa11b1629/bench/bench.cpp#L96-L131).
During index construction the hash function is hidden behind a unique pointer, and the vector to be hashed as well.
This indirection is rather expensive, as shown in the table below, were we compare this invocation pattern with the invocation without virtual calls.
Again, the times are in nanoseconds.

|               ns/op |                op/s |    err% |     total | Hashing
|--------------------:|--------------------:|--------:|----------:|:--------
|              348.17 |        2,872,187.65 |    0.0% |      0.00 | `FHT cross polytope`
|            1,021.79 |          978,674.22 |    0.8% |      0.00 | `FHT cross polytope (indirection)`
|                5.02 |      199,068,622.20 |    0.1% |      0.00 | `SimHash`
|              254.47 |        3,929,774.42 |    0.3% |      0.00 | `SimHash (indirection)`
