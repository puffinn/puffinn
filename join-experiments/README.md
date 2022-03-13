# Join experiments

This directory contains experiments for both the top-k join and the k-nn join (aka k-nn graph construction).

## Top-k join

### Baselines

The source `XiaoEtAl.cpp` contains an implementation of the algorithm from the paper
> C. Xiao, W. Wang, X. Lin and H. Shang,
> "Top-k Set Similarity Joins,"
> 2009 IEEE 25th International Conference on Data Engineering, 2009,
> pp. 916-927, doi: 10.1109/ICDE.2009.111.

After compiling, the invocation is as follows:

    XiaoEtAl file topk

where `topk` is the requested number of top-k pairs.
The data file is a text file, where the first line reports the universe size (i.e. the size of the union of all sets in the dataset).
All subsequent lines report elements encoded as integers, with the understanding that smaller numbers correspond to less frequent tokens.
The script `dblp.py` produces such a file, using author names and paper titles from [DBLP](https://dblp.uni-trier.de/).

The table below reports some running times for different values of k, along with the similarity of the k-th pair.

| dataset |    k | elapsed (ms) | similarity |
| :------ | ---: | -----------: | ---------: |
| dblp.vecs.txt | 10 | 4465 | 0.966667 |
| dblp.vecs.txt | 100 | 4672 | 0.952381 |
| dblp.vecs.txt | 1000 | 7097 | 0.928571 |
| dblp.vecs.txt | 10000 | 72652 | 0.833333 |
