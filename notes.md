#### Questions
- Which data format should we use? unit\_vector cannot store values larger than 1, which is needed for average vectors and such
- How do we even compile and run the code correctly
- How should we choose M? (Multiple of # of elements in avx2?), or something else with padding inside vectors
- Inertia can overflow? Should we preprocess vectors?



#### TODO:
- get Global minimum Enertia not local


Format for PQ:
- input: Dataset  
- Construct lookup table
- input: query vector
- output: whatever paper says, probably indexes for data entries closest to query

