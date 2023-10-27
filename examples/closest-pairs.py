import sys
from puffinn import *
import numpy
import time
import math

import pickle
def angular_dist(a, b):
    return numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))

MB = 1024*1024

dimensions = 100
n = 1000
k = 10

# Generate random data and compute the ground truth.
print('Creating %d %d-dimensional points' % (n, dimensions))
dataset = [[numpy.random.normal(0, 1) for _ in range(dimensions)]  for _ in range(n)]

print('Computing ground truth')
ground_truth = [
    sorted([(angular_dist(q, v), i, j)
            for (i, v) in enumerate(dataset) if i < j])[-k:] 
    for (j, q) in enumerate(dataset)
]
ground_truth = sorted([tup for topk in ground_truth for tup in topk])[-k:]
for g in ground_truth:
    print(g)

# Construct the search index.
# Here we use angular distance aka cosine similarity
# with the default hash functions and 500MB memory.
index = Index('angular', dimensions, 500*MB)
print('Building index')
for v in dataset:
    index.insert(v)

t0 = time.time()
index.rebuild()
print("Building index took %.2f seconds." % (time.time() - t0) )

# The index can be stored and loaded using pickle.
serialized = pickle.dumps(index)
index = pickle.loads(serialized)

results = []
print('Searching the index for the closest pairs')
t0 = time.time()
results = index.closest_pairs(k, 0.8)

print("Search the index took %.2f seconds." % (time.time() - t0))

found = 0
dresults = []
for i, j in results:
    d = angular_dist(dataset[i], dataset[j])
    if d >= ground_truth[0][0]:
        found += 1
    dresults.append(
        (d, i, j)
    )
dresults = sorted(dresults)
for d in dresults:
    print(d)

print("recall: %f" % (found / k))

