import numpy
import sys

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <groundtruth> <exp-results>")
    exit(1)

gt = sys.argv[1]
actual = sys.argv[2]

def parsefile(f):
    n, k = numpy.fromfile(f, dtype="uint64", count=2)
    arr = numpy.fromfile(f, dtype="uint32", offset=16).reshape(n, k)
    return arr

def compute_recall(arr1, arr2):
    n, k = arr1.shape
    expected = n * k
    found = 0
    for i in range(arr1.shape[0]):
        found += len(numpy.intersect1d(arr1[i], arr2[i]))
    return found / expected


gt = parsefile(gt)
ac = parsefile(actual)

print(f"Recall: {compute_recall(gt, ac)}")

