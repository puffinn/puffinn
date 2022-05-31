import h5py
import numpy as np

def compute_distances():
    f = h5py.File(h5py_path)
    assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
    self.data = np.array(f['train'])
    norms = np.linalg.norm(self.data, axis=1)[:, np.newaxis]
    assert np.sum(norms == 0.0) == 0
    self.data /= norms
    f.close()

def compute_lid(k, distances):
    estimates = []
    
    assert distances.shape[1] >= k, "Not enough distances to compute LID@k".format(k)

    for i, vec in enumerate(distances):
        vec.sort()
        w = vec[k]
        half_w = 0.5 * w

        vec = vec[:k+1]
        vec = vec[vec > 1e-5]

        small = vec[vec < half_w]
        large = vec[vec >= half_w]

        s = numpy.log(small / w).sum() + numpy.log1p((large - w) / w).sum()
        valid = small.size + large.size
        
        estimates.append(-valid / s)
        
    return np.array(estimates)
