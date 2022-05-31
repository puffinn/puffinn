# Compute measures such as LID and RC for datasets

import os
import numpy as np
import h5py
import sys
import faiss

DIR_ENVVAR = 'TOPK_DIR'
try:
    BASE_DIR = os.environ[DIR_ENVVAR]
except:
    print("You should set the {} environment variable to a directory".format(DIR_ENVVAR))
    sys.exit(1)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULT_FILES_DIR = os.path.join(BASE_DIR, "output")

def compute_dists(k, dataset, distancefn):
    if distancefn == 'angular' or distancefn == 'cosine':
        norms = np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        assert np.sum(norms == 0.0) == 0
        dataset /= norms
        index = faiss.IndexFlatIP(dataset.shape[1])
        index.add(dataset)
        all_distances = index.search(dataset, k+1)[0]
        avg_distances = np.mean(all_distances, axis=1)
        distances = all_distances[:, 1:]
        return distances, avg_distances
    else:
        raise Exception("unsupported similarity measure {}".format(distancefn))
    
def compute_lids(k, distances):
    assert distances.shape[1] >= k
    estimates = []

    for vec in distances:
        vec.sort()
        w = vec[k]
        half_w = 0.5 * w

        # Use numpy vector operations to improve efficiency.
        # Results are the same up the the 6th decimal position
        # compared to iteration
        vec = vec[:k]
        vec = vec[vec > 1e-5]

        small = vec[vec < half_w]
        large = vec[vec >= half_w]

        s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
        valid = small.size + large.size

        estimates.append(-valid / s)

    return np.array(estimates)

def compute_relative_contrasts(k, distances, avg_distances):
    return avg_distances / distances[:, k]

if __name__ == "__main__":
    path = sys.argv[1]
    k = 10
    kdists = 1000
    with h5py.File(path, 'r+') as hfp:
        dkey = '/top-{}-dists'.format(kdists)
        avg_dist_key = '/avg-dists'.format(kdists)
        if not dkey in hfp:
            dataset = np.array(hfp['/train'][:10000,:]).astype(np.float32)
            distances, avg_distances = compute_dists(kdists, dataset, hfp.attrs['distance'])
            hfp[dkey] = distances
            hfp[avg_dist_key] = avg_distances
        distances = hfp[dkey]
        avg_distances = hfp[avg_dist_key]

        lid_key = '/lid-at-{}'.format(k)
        if not lid_key in hfp:
            hfp[lid_key] = compute_lids(k, distances)
        
        rc_key = '/rc-at-{}'.format(k)
        if not rc_key in hfp:
            compute_relative_contrasts(k, distances, avg_distances)

