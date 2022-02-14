#### Questions
- Which data format should we use? unit\_vector cannot store values larger than 1, which is needed for average vectors and such
- How do we even compile and run the code correctly
- How should we choose M? (Multiple of # of elements in avx2?), or something else with padding inside vectors
- Inertia can overflow? Should we preprocess vectors?
 
#### Readings
Both [1] does some precomputation by either random permutation or multiplication with an orthonormal matrix, respectively. Why is this?
Seems many approches require linear algebra, shouldn't we use a library? (This would create issues with the currently implemented Dataset storing format)

[1] - Quantization based Fast Inner Product Search



Format for PQ:
- input: Dataset  
- Construct lookup table
- input: query vector
- output: Estimated cosine similarity
- Follow that of the filter already inside puffin that is used to store sketches
- We have to determine threshholds for when data entries are added to the buffer, at least for the sketch distances, not sure how we will handle the stopping criterion

#### TODO:
- [ ] Begin implementation of naive PQ with both euclidean dist optimization and mahalanobis dist. ~ 
- [ ] Implement random permutation of data points, does it have any effect on (LSH scheme?)
- [x] Working implementation of PQ class to follow format of 'filterer' in index class, and decide design for codebook. i.e. 
- [x] Implement simple PQ Code function
- [ ] deciding sizes when $d/M \mod 2 \ne 0$ (**VIKTOR**)
- [ ] Get SIMD to work for subspaces
- [ ] Begin writing related work for original PQ paper and litterature related to that as well (llyod algo).
- [ ] Test quantization error
- [ ] Begin writing formal problem definition of ANN
- [ ] Create quick testing setup using acutal data (Investigate if ANN-Benchmark can be used through small datasample and only 1 not all datasets) (**TIM**)
-[] Look at previous bsc. projects of what is included and to what level of expertize.



#### Agenda for next meeting
- Explain readings and what we think is important and should be implemented
- Our next steps i.e. TODO list
- Figuring out the threshhold for adding to the buffer?
    - Empirically figure out what works at 'index building time'
    - Bootstrap threshhold such that X\% is above threshhold according to real inner products (maybe faulty as estimated cosine dists are biased??)
    - Other options?


