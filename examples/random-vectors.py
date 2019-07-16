from puffinn import *
import numpy
import time
import math

def angular_dist(a, b):
    return numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))

MB = 1024*1024

dimensions = 100
n = 10000
k = 10
n_queries = 100

# Generate random data and compute the ground truth.
print('Creating %d %d-dimensional points and %d queries' % (n, dimensions, n_queries))
dataset = [[numpy.random.normal(0, 1) for _ in range(dimensions)]  for _ in range(n)]
queries = [[numpy.random.normal(0, 1) for _ in range(dimensions)] for _ in range(n_queries)]

print('Computing ground truth')
ground_truth = [
    sorted([angular_dist(q, v) for v in dataset])[-10:] for q in queries]

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

results = []
print('Searching the index')
t0 = time.time()
for query in queries:
    # Search the index for the k closest points.
    # Each of the k closest points have at least an 80% chance of being found.
    results.append(index.search(query, k, 0.8))

print("Search the index took %.2f seconds." % (time.time() - t0))

found = 0
for i, r in enumerate(results):
    found += len([j for j in r if angular_dist(queries[i], dataset[j]) >= ground_truth[i][0]])

print('Average recall: %f' % (found / (k * n_queries)))









