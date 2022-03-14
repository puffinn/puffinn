import h5py
import numpy
import sys

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <groundtruth> <exp-results>")
    exit(1)

gt = sys.argv[1]
actual = sys.argv[2]

def parsefile(f):
    f = h5py.File(f, "r")
    return numpy.array(f['results']), f.attrs['time']

def compute_recall(arr1, arr2):
    n, k = arr1.shape
    expected = n * k
    found = 0
    for i in range(arr1.shape[0]):
        found += len(numpy.intersect1d(arr1[i], arr2[i]))
    return found / expected


gt, tgt = parsefile(gt)
ac, tac = parsefile(actual)

print(f"Recall: {compute_recall(gt, ac)}")
print(f"Speedup: {tgt/tac}")

