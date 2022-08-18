#!/usr/bin/env python3

# This script handles the execution of the join experiments
# in two different modes:
#  - global top-k
#  - local top-k
#
# Datasets are taken from ann-benchmarks or created ad-hoc from
# other sources (e.g. DBLP).

import concurrent.futures
import gzip
import urllib.request
import zipfile
from tqdm import tqdm
import numpy as np
import time
import sklearn
import sklearn.preprocessing
import subprocess
import h5py
import sys
import yaml
import shlex
import os
import hashlib
import sqlite3
import faiss
import falconn
import json
import random
import numba
import heapq
import copy
import pynndescent
import scipy
from urllib.request import urlopen
from urllib.request import urlretrieve
from pprint import pprint


DIR_ENVVAR = 'TOPK_DIR'
try:
    BASE_DIR = os.environ[DIR_ENVVAR]
except:
    print("You should set the {} environment variable to a directory".format(DIR_ENVVAR))
    sys.exit(1)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULT_FILES_DIR = os.path.join(BASE_DIR, "output")

# Results database
# ================
#
# We store results in a Sqlite database, which points to paths to HDF5 files
# that store the actual nearest neighbor information, which would be too large
# to store in the sqlite database.
# From these HDF5 files we compute the recall of the various algorithms, which
# are then cached in the database for fast query.

# This is a sequence of SQL statements that set up different
# versions of the database.
MIGRATIONS = [
    """
    CREATE TABLE main (
        dataset            TEXT NOT NULL,
        workload           TEXT NOT NULL,
        k                  INTEGER NOT NULL,
        algorithm          TEXT NOT NULL,
        params             TEXT NOT NULL,
        threads            INT,
        time_index_s       REAL NOT NULL,
        time_join_s        REAL NOT NULL,
        recall             REAL, -- may be null, we compute it afterwards
        output_file        TEXT NOT NULL,
        hdf5_group         TEXT NOT NULL
    )
    """,
    """
    CREATE VIEW baselines AS
    SELECT * FROM main WHERE algorithm = 'BruteForceLocal'
    """,
    """
    CREATE VIEW baselines_global AS
    SELECT * FROM main WHERE algorithm = 'XiaoEtAl'
    """,
    """
    ALTER TABLE main ADD COLUMN algorithm_version INT DEFAULT 1;
    """,
    """
    CREATE VIEW recent_versions AS
    SELECT algorithm, max(algorithm_version) as algorithm_version
    FROM main
    GROUP BY 1;

    CREATE VIEW recent AS
    SELECT *
    FROM main
    NATURAL JOIN recent_versions;
    """
]

def get_db():
    db = sqlite3.connect(os.path.join(BASE_DIR, "join-results.db"), isolation_level=None)
    current_version, = db.execute("pragma user_version;").fetchone()
    # Update the database schema, if needed
    for i, migration in enumerate(MIGRATIONS):
        version = i + 1
        if version > current_version:
            db.executescript(migration)
            db.execute("pragma user_version = {}".format(version))
    return db


def already_run(db, configuration):
    """Checks whether the given configuration is already present in the database"""
    configuration = configuration.copy()
    algorithm_version = ALGORITHMS[configuration['algorithm']]()[1]
    configuration['threads'] = configuration.get('threads', 1)
    configuration['params']['threads'] = configuration.get('threads', 1)
    configuration['params'] = json.dumps(configuration['params'], sort_keys=True)
    print("ALREADY RUN?", configuration)
    configuration['algorithm_version'] = algorithm_version
    res = db.execute("""
    SELECT rowid FROM main
    WHERE dataset = :dataset
      AND workload = :workload
      AND threads = :threads
      AND k = :k
      AND algorithm = :algorithm
      AND algorithm_version = :algorithm_version
      AND params = :params
    """, configuration).fetchall()
    return len(res) > 0

def compute_recall(k, baseline_indices, actual_indices, output_file=None, hdf5_group=None):
    baseline_indices = baseline_indices[:,:k]
    assert baseline_indices.shape[1] == actual_indices.shape[1]
    # If we have fewer rows we used a prefix for the evaluation
    assert baseline_indices.shape[0] <= actual_indices.shape[0]
    print(
        "Indices",
        baseline_indices[0],
        actual_indices[0],
        sep="\n"
    )
    recalls = np.array([
        np.mean(np.isin(baseline_indices[i], actual_indices[i]))
        for i in tqdm(range(len(baseline_indices)), leave=False)
    ])
    if output_file is not None:
        with h5py.File(output_file, 'r+') as hfp:
            gpath = 'local-top-{}-recalls'.format(k)
            if gpath in hfp[hdf5_group]:
                print('deleting existing recall matrix', gpath)
                del hfp[hdf5_group][gpath]
            hfp[hdf5_group][gpath] = recalls
    avg_recall = np.mean(recalls)
    return avg_recall


@numba.jit
def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
         return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)


def compute_distances(k, dataset, distancefn):
    if distancefn == 'angular' or distancefn == 'cosine':
        norms = np.linalg.norm(dataset, axis=1)[:, np.newaxis]
        assert np.sum(norms == 0.0) == 0
        dataset /= norms
        index = faiss.IndexFlatIP(dataset.shape[1])
        index.add(dataset)
        all_distances, all_neighbors = index.search(dataset, k+1)
        avg_distances = np.mean(all_distances, axis=1)
        distances = all_distances[:, 1:]
        neighbors = all_neighbors[:, 1:]
        return distances, neighbors, avg_distances
    if distancefn == 'jaccard':
        from scipy.spatial.distance import pdist
        dataset = np.array(dataset)
        print(dataset)
        distances = pdist(np.array(dataset), jaccard)
        print(distances)
    else:
        raise Exception("unsupported similarity measure {}".format(distancefn))


def get_baseline_indices(db, dataset, k):
    with h5py.File(DATASETS[dataset]()) as hfp:
        # Read the neighbors directly from the dataset file, if those are available
        if 'top-1000-neighbors' in hfp:
            return hfp['top-1000-neighbors'][:,:k]
    baseline = db.execute(
        "SELECT output_file, hdf5_group FROM baselines WHERE dataset = :dataset AND workload = 'local-top-k';",
        {"dataset": dataset}
    ).fetchone()
    if baseline is None:
        print("Missing baseline")
        return None
    base_file, base_group = baseline
    base_file = os.path.join(BASE_DIR, base_file)
    with h5py.File(base_file) as hfp:
        baseline_indices = hfp[base_group]['local-top-1000'][:,:k]
        return baseline_indices

def compute_recalls(db):
    # Local topk
    missing_recalls = db.execute("SELECT rowid, algorithm, params, dataset, k, output_file, hdf5_group FROM main WHERE recall IS NULL AND WORKLOAD = 'local-top-k' and k <= 10;").fetchall()
    print("there are {} missing recalls for local topk".format(len(missing_recalls)))
    for rowid, algorithm, params, dataset, k, output_file, hdf5_group in missing_recalls:
        if k > 10: # FIXME handle the case for k > 10 instead of skipping it
            continue
        print("Computing recalls for {} {} on {} with k={}".format(algorithm, params, dataset, k))
        baseline_indices = get_baseline_indices(db, dataset, k)
        if baseline_indices is None:
            print("Missing baseline")
            continue
        output_file = os.path.join(BASE_DIR, output_file)
        print("output file", output_file, "rowid", rowid, "group", hdf5_group)
        with h5py.File(output_file) as hfp:
            actual_indices = np.array(hfp[hdf5_group]['local-top-{}'.format(k)])
        avg_recall = compute_recall(k, baseline_indices, actual_indices, output_file, hdf5_group)
        print("Average recall is", avg_recall)
        db.execute(
            """UPDATE main
                 SET recall = :recall
               WHERE rowid = :rowid ;
            """,
            {"rowid": rowid, "recall": avg_recall}
        )
    return

    # Global topk
    missing_recalls = db.execute("SELECT rowid, algorithm, params, dataset, k, output_file, hdf5_group FROM main WHERE recall IS NULL AND WORKLOAD = 'global-top-k';").fetchall()
    print("There are {} missing recalls for global-top-k".format(len(missing_recalls)))
    for rowid, algorithm, params, dataset, k, output_file, hdf5_group in missing_recalls:
        # Compute the top-1000 distances for the dataset, if they are not already there
        dist_key, nn_key = '/top-1000-dists', '/top-1000-neighbors'
        top_pairs_key = '/top-1000-pairs'

        dataset_path = DATASETS[dataset]()
        with h5py.File(dataset_path, 'r+') as hfp:
            if dist_key not in hfp or nn_key not in hfp:
                print('Computing top distances for', dataset)
                distances, neighbors, avg_distance = compute_distances(1000, hfp['/train'], hfp.attrs['distance'])
                hfp[dist_key] = distances
                hfp[nn_key] = neighbors
                hfp['/average_distance'] = avg_distance
            if top_pairs_key not in hfp:
                print('Computing top 1000 pairs for', dataset)
                distances = hfp[dist_key]
                neighbors = hfp[nn_key]
                topk = []
                for i, (dists, neighs) in tqdm(enumerate(zip(distances, neighbors)), total=neighbors.shape[0]):
                    for d, j in zip(dists, neighs):
                        if i != j:
                            t = (d, min(i, j), max(i, j))
                            if len(topk) > 2000:
                                heapq.heappushpop(topk, t)
                            else:
                                heapq.heappush(topk, t)
                topk = list(set(topk)) # remove duplicates
                topk.sort(reverse=True)
                topk = topk[:1000]
                hfp[top_pairs_key] = topk

            baseline_pairs = set([(min(pair[0], pair[1]), max(pair[0], pair[1])) for pair in hfp[top_pairs_key][:k, 1:3].astype(np.int32)])
            baseline_dists = hfp[top_pairs_key][:k, 0]


        print("Computing recalls for {} {} on {} with k={}".format(algorithm, params, dataset, k))
        print(baseline_pairs)
        print(baseline_dists)
        output_file = os.path.join(BASE_DIR, output_file)
        with h5py.File(output_file) as hfp:
            actual_pairs = set(map(tuple, hfp[hdf5_group]['global-top-{}'.format(k)]))
        print("Actual pairs")
        print(actual_pairs)
        matched = 0
        for pair in baseline_pairs:
            if pair in actual_pairs:
                matched += 1
        recall = matched / len(baseline_pairs)
        db.execute(
            """UPDATE main
                 SET recall = :recall
               WHERE rowid = :rowid ;
            """,
            {"rowid": rowid, "recall": recall}
        )




# =============================================================================
# Algorithms
# ==========
#
# Communication protocol
# ----------------------
#
# For algorithms interfacing over text streams, the protocol is as follows:
#
# - Data ingestion
#   - harness sends `sppv2 data`, `sspv1 distance_type`, followed by vectors, one per line, followed by `sppv2 end`
#   - program acknowledges using `sppv2 ok`
# - Index construction (timed)
#   - harness sends `sppv2 index param1 v1 param2 v2 ...`
#   - program acknowledges using `sppv2 ok` when it's done
# - Workload run (timed)
#   - harness sends `sppv2 workload param1 v1 param2 v2`
#   - program acknowledges using `sppv2 ok` when it's done
# - Result collection
#   - harness sends `sppv2 result`
#   - program writes output on stdout, one item per line (for whatever "item" means for the workload)
#   - program writes `sppv2 end`
# - Iteration or stop
#   - harness sends either `sppv2 end_workloads` to signal that we are done or
#   - sends a new workload specification

PROTOCOL = "sppv2" # Version of the communication protocol


def text_encode_floats(v):
    return " ".join(map(str, v))

def text_encode_ints(v):
    return " ".join(map(str, map(int, v)))

def h5cat(path, stream=sys.stdout):
    print('catting cata from', path)
    file = h5py.File(path, "r")
    distance = file.attrs['distance']
    if distance == 'cosine' or distance == 'angular':
        for v in tqdm(file['train'], leave=False):
            v = v / np.linalg.norm(v)
            stream.write(text_encode_floats(v) + "\n")
            stream.flush()
            # print(text_encode_floats(v), file=stream)
        print(file=stream) # Signal end of streaming
    elif distance == 'jaccard':
        data = np.array(file['train'])
        sizes = np.array(file['size_train'])
        offsets = np.zeros(sizes.shape, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes[:-1])
        for offset, size in tqdm(zip(offsets, sizes)):
            v = data[offset:offset+size]
            if len(v) > 0:
                txt = text_encode_ints(v)
                stream.write(txt + "\n")
                stream.flush()
        print(file=stream)
    else:
        raise RuntimeError("Unsupported distance".format(distance))


class Algorithm(object):
    """Manages the lifecycle of an algorithm"""
    def execute(self, k, params, h5py_path, output_file, output_hdf5_path):
        # self.setup(k, params)
        # self.feed_data(h5py_path)
        # self.index()
        # self.run()
        # self.save_result(output_file, output_hdf5_path)
        # return self.times()
        raise Exception("deprecated")
    def setup(self, k):
        """Configure the parameters of the algorithm"""
        raise Exception("deprecated")
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        raise Exception("not implemented")
    def index(self, params):
        """Setup the index with the given parameters, if any. This is timed."""
        raise Exception("not implemented")
    def run(self, params):
        """Run the workload with the given join params. This is timed."""
        raise Exception("not implemented")
    def clear(self):
        """Clears the result of the previous run"""
        raise Exception("not implemented")
    def result(self):
        """Return the result as a two-dimensional numpy array.
        The way it is interpreted depends on the application.
        For global top-k join, it is the list of top-k pairs of indices.
        For local top-k join, it is the list of
        nearest neighbors for each element.
        """
        raise Exception("not implemented")
    def save_result(self, hdf5_file, path):
        result = self.result()
        if path in hdf5_file:
            del hdf5_file[path]
        #     existing = np.array(hdf5_file[path])
        #     print(existing)
        #     assert (existing == result).all()
        #     return
        hdf5_file[path] = result
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        raise Exception("not implemented")
    def done(self):
        """Used to signal to implementations that we are done and that the index can be discarded"""
        pass


class PyNNDescent(Algorithm):
    def __init__(self):
        self._query_matrix = None
        self.data = None
        self.metric = None
        self._index = None

    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""

        distance = h5py.File(h5py_path).attrs['distance']
        self.metric = distance
        if distance == "angular":
            distance = "cosine"

        if distance == "jaccard":
            # Load sparse data into a sparse matrix.
            X = list(stream_sparse(h5py_path))
            sizes = [len(x) for x in X]
            n_cols = max([max(x) for x in X]) + 1
            matrix = scipy.sparse.csr_matrix((len(X), n_cols), dtype=np.float32)
            matrix.indices = np.hstack(X).astype(np.int32)
            matrix.indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
            matrix.data = np.ones(matrix.indices.shape[0], dtype=np.float32)
            matrix.sort_indices()
            X = matrix
        else:
            X = h5py.File(h5py_path)['train'][:]
        self.data = X

    def index(self, params):
        """Setup the index with the given parameters, if any. This is timed."""
        start = time.time()
        self._index = pynndescent.NNDescent(
            self.data,
            n_neighbors=params['n_neighbors'],
            metric=self.metric,
            low_memory=True,
            leaf_size=params['leaf_size'],
            pruning_degree_multiplier=params['pruning_degree_multiplier'],
            diversify_prob=params['diversify_prob'],
            # n_trees=params['n_trees'],
            # n_search_trees=params['n_search_trees'],
            compressed=False,
            verbose=True
        )
        end = time.time()
        self.index_time = end - start

    def run(self, params):
        """Run the workload with the given join params. This is timed."""
        k = params['k']
        start = time.time()
        # Since we are doing a self-join we can and should use the graph
        # that is built into the index
        ind = self._index.neighbor_graph[0][:,1:(k+1)]
        end = time.time()
        self.workload_time = end - start
        self.result_indices = ind

    def clear(self):
        """Clears the result of the previous run"""
        self.workload_time = None
        self.result_indices = None

    def result(self):
        """Return the result as a two-dimensional numpy array.
        The way it is interpreted depends on the application.
        For global top-k join, it is the list of top-k pairs of indices.
        For local top-k join, it is the list of
        nearest neighbors for each element.
        """
        return self.result_indices

    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.index_time, self.workload_time


class SubprocessAlgorithm(Algorithm):
    """Manages the lifecycle of an algorithm which does not provide
    a Python interface"""

    def _raw_line(self):
        if self._subprocess_handle().poll():
            raise Exception('Subprocess killed')
        return shlex.split(
            self._subprocess_handle().stdout.readline().strip())

    def _line(self):
        line = self._raw_line()
        while len(line) < 1 or line[0] != PROTOCOL:
            line = self._raw_line()
        return line[1:]

    def _send(self, msg):
        program = self._subprocess_handle()
        print(PROTOCOL, msg, file=program.stdin)

    def _expect(self, what, errmsg):
        assert self._line()[0] == what, errmsg

    # param result_collector: object that accepts lines with a method `add_line`,
    # one by one, and converts them to at global or local top-k result
    def __init__(self, command_w_args, profile=False):
        self.command_w_args = command_w_args
        self._program = None
        self.index_time = None
        self.workload_time = None
        self.profile = profile

    def _subprocess_handle(self):
        if self._program is None:
            cmdline = self.command_w_args if not self.profile else ['flamegraph', '--root', '--'] + self.command_w_args
            self._program = subprocess.Popen(
                cmdline,
                bufsize=1,  # line buffering
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True)
        return self._program

    def _wait_for_completion(self):
        self._subprocess_handle().wait()
        self._program = None

    # def setup(self, k, params_dict):
    #     print("Setup")
    #     self._send("setup")
    #     program = self._subprocess_handle()
    #     print("k", k, file=program.stdin)
    #     for key, v in params_dict.items():
    #         print(key, v, file=program.stdin)
    #     self._send("end")
    #     self._expect("ok", "setup failed")

    def feed_data(self, h5py_path):
        distance = h5py.File(h5py_path).attrs['distance']
        if distance == "angular":
            distance = "cosine"
        self._send("data")
        self._send(distance)
        program = self._subprocess_handle()
        self._send("path " + h5py_path)
        self._expect("ok", "population phase failed")

    def index(self, params):
        print("Building index")
        start_t = time.time()
        cmd_string = "index " + " ".join(["{} {}".format(k, v) for k, v in params.items()])
        self._send(cmd_string)
        self._expect("ok", "workload phase failed")
        end_t = time.time()
        self.index_time = end_t - start_t

    def run(self, params):
        print("Running workload")
        start_t = time.time()
        cmd_string = "workload " + " ".join(["{} {}".format(k, v) for k, v in params.items()])
        self._send(cmd_string)
        # Wait for the program to report success
        self._expect("ok", "workload phase failed")
        end_t = time.time()
        self.workload_time = end_t - start_t

    def result(self):
        print("Running collecting result")
        self._send("result")
        rows = []
        while True:
            line = self._raw_line()
            if line[-1] == "end":
                break
            row = np.array([int(i) for i in line])
            assert np.unique(row).shape == row.shape
            rows.append(row)
        rows = np.array(rows)
        return rows

    def done(self):
        self._send("end_workloads")
        print("Waiting for subprocess to finish")
        self._wait_for_completion()

    def clear(self):
        pass # nothing to do here

    def times(self):
        return self.index_time, self.workload_time


class FaissIVF(Algorithm):

    def __init__(self):
        self.faiss_index = None
        self.k = None
        self.data = None
        self.time_index = None
        self.time_run = None
        self.result_indices = None
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train'])
        f.close()
    def index(self, params):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()
        faiss.omp_set_num_threads(params['threads'])

        n_list = params["n_list"]

        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        self.faiss_index = index
        self.time_index = time.time() - start
    def run(self, params):
        """Run the workload. This is timed."""
        print("  Top-{} join".format(self.k))
        k = params['k']
        start = time.time()
        n_probe = params["n_probe"]
        index.nprobe = n_probe
        _dists, idxs = self.faiss_index.search(self.data, k+1)
        self.time_run = time.time() - start
        self.result_indices = idxs[:,1:]
    def result(self):
        return self.result_indices
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.time_index, self.time_run
    def clear(self):
        self.result_indices = None
        self.time_run = None


class FaissHNSW(Algorithm):
    def __init__(self):
        self.faiss_index = None
        self.k = None
        self.data = None
        self.time_index = None
        self.time_run = None
        self.result_indices = None
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train']).astype(np.float32)
        f.close()
    def index(self, params):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()
        faiss.omp_set_num_threads(params['threads'])
        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.faiss_index = faiss.IndexHNSWFlat(len(X[0]), params["M"])
        self.faiss_index.hnsw.efConstruction = params["efConstruction"]
        self.faiss_index.add(X)
        self.time_index = time.time() - start
    def run(self, params):
        """Run the workload. This is timed."""
        print("  Top-{} join".format(self.k))
        self.faiss_index.hnsw.efSearch = params["efSearch"]
        start = time.time()
        k = params['k']
        _dists, idxs = self.faiss_index.search(self.data, k+1)
        self.time_run = time.time() - start
        self.result_indices = idxs[:,1:]
    def result(self):
        return self.result_indices
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.time_index, self.time_run
    def clear(self):
        self.result_indices = None
        self.time_run = None


class FALCONN(Algorithm):

    def __init__(self):
        self._index = None
        self.k = None
        self.data = None
        self.time_index = None
        self.time_run = None
        self.result_indices = None

    def clear(self):
        self.result_indices = None
        self.time_run = None

    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train'])
        f.close()

    def index(self, params):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()

        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = X.shape[1]
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params_cp.storage_hash_table = falconn.StorageHashTable.FlatHashTable
        params_cp.k = params["k"]
        params_cp.l = params["L"]
        params_cp.num_setup_threads = 0
        params_cp.last_cp_dimension = 16
        params_cp.num_rotations = 3
        params_cp.seed = 833840234

        self._index = falconn.LSHIndex(params_cp)
        self._index.setup(X)
        self.time_index = time.time() - start

    def _run_individual_query(self, query, params):
        qo = self._index.construct_query_object()
        qo.set_num_probes(params["num_probes"])
        res = qo.find_k_nearest_neighbors(query, self.k + 1)
        if len(res) < self.k + 1:
            res += [0] * (self.k + 1 - len(res))
        return res

    def run(self, params):
        """Run the workload. This is timed."""
        print("  Top-{} join".format(self.k))
        start = time.time()
        k = params['k']
        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.result_indices = np.zeros((X.shape[0], k + 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=params["threads"]) as executor:
            tasks = {executor.submit(self._run_individual_query, query, params): i for (i, query) in enumerate(X)}
            for future in concurrent.futures.as_completed(tasks):
                i = tasks[future]
                res = future.result()
                self.result_indices[i] = res

        #for (i, query) in enumerate(X):
        #    self.result_indices[i] = self._run_individual_query(query)

        self.time_run = time.time() - start
        self.result_indices = self.result_indices[:,1:]

    def result(self):
        return self.result_indices

    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.time_index, self.time_run


class BruteForceLocal(Algorithm):
    def __init__(self):
        self.data = None
        self.time_index = 0
        self.time_run = None
        self.result_indices = None
    # def setup(self, k, params):
    #     self.k = k
    #     self.prefix = params.get('prefix')
    #     pass
    def feed_data(self, h5py_path):
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train']).astype(np.float32)
        norms = np.linalg.norm(self.data, axis=1)[:, np.newaxis]
        assert np.sum(norms == 0.0) == 0
        self.data /= norms
        f.close()
    def clear(self):
        # Nothing to do here
        pass
    def index(self, params):
        self.time_index = 0
    def run(self, params):
        k = params['k']
        print("  Top-{} join".format(k))
        start = time.time()
        index = faiss.IndexFlatIP(len(self.data[0]))
        index.add(self.data)
        if 'prefix' not in params:
            queries = self.data
        else:
            queries = self.data[0:params['prefix']]
            print(queries)
        self.result_indices = index.search(queries, k+1)[1][:, 1:]
        print(self.result_indices)
        self.time_run = time.time() - start
    def result(self):
        return self.result_indices
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.time_index, self.time_run


# =============================================================================
# Datasets
# ========
#
# Here we define preprocessing code for datasets, that will also fetch
# them if already available. Each function returns the local path to the
# preprocessed dataset.

def download(url, dest):
    if not os.path.isfile(dest):
        print("Downloading {} to {}".format(url, dest))
        urllib.request.urlretrieve(url, dest)


def write_sparse(out_fn, data):
    f = h5py.File(out_fn, 'w')
    f.attrs['distance'] = 'jaccard'

    data = np.array(list(map(sorted, data)))
    flat_data = np.hstack(data.flatten())
    f.create_dataset('train', (len(flat_data),), dtype=flat_data.dtype)[:] = flat_data
    f.create_dataset('size_train', (len(data),), dtype='i')[:] = list(map(len, data))
    f.close()

def write_dense(out_fn, vecs):
    hfile = h5py.File(out_fn, "w")
    hfile.attrs["dimensions"] = len(vecs[0])
    hfile.attrs["type"] = "dense"
    hfile.attrs["distance"] = "cosine"
    hfile.create_dataset("train", shape=(len(vecs), len(vecs[0])), dtype=vecs[0].dtype, data=vecs, compression="gzip")
    hfile.close()

def stream_sparse(in_fn):
    file = h5py.File(in_fn)
    data = np.array(file['train'])
    sizes = np.array(file['size_train'])
    offsets = np.zeros(sizes.shape, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    for offset, size in zip(offsets, sizes):
        v = data[offset:offset+size]
        if len(v) > 0:
            yield v

# Adapted from ann-benchmarks
def random_jaccard(out_fn, n=10000, size=50, universe=80):
    if os.path.isfile(out_fn):
        return out_fn
    random.seed(1)
    l = list(range(universe))
    # We call the set of sets `train` to be compatible with datasets from
    # ann-benchmarks
    train = []
    for i in range(n):
        train.append(random.sample(l, size))

    write_sparse(out_fn, train)
    return out_fn

def random_float(out_fn, n_dims, n_samples, centers):
    import sklearn.datasets
    if os.path.isfile(out_fn):
        return out_fn

    X, _ = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=n_dims,
        centers=centers, random_state=1)
    X = X.astype(np.float32)
    write_dense(out_fn, X)
    return out_fn

def random_difficult(out_fn, n, d, k):
    if os.path.isfile(out_fn):
        return out_fn
    
    assert d % 3 == 0
    assert n % (k + 1) == 0

    D = d // 3
    # the base set of random vectors, expected unit length
    np.random.seed(1)
    X = np.random.normal(scale=1/D, size=(n// (k + 1), d))

    Y = np.zeros((n, d))
    for i, x in enumerate(X):
        Y[(k + 1) * i] = x
        for j in range(1, k + 1):
            # k random vectors at expected distance \sqrt{1/3} of x
            Y[(k + 1) * i + j] = np.concatenate((x[: 2 * D], np.random.normal(scale=1/D, size=D)))
    Y = Y.astype(np.float32)
    write_dense(out_fn, Y)
    return out_fn


def glove(out_fn, dims):
    if os.path.isfile(out_fn):
        return out_fn
    url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    localzip = os.path.join(DATASET_DIR, "glove.twitter.27B.zip")
    download(url, localzip)

    with zipfile.ZipFile(localzip) as zp:
        z_fn = 'glove.twitter.27B.%dd.txt' % dims
        vecs = np.array([
            np.array([float(t) for t in line.split()[1:]])
            for line in zp.open(z_fn)
        ])
        write_dense(out_fn, vecs)
    return out_fn

# Adapted from https://github.com/erikbern/ann-benchmarks.
# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn):
    if os.path.isfile(out_fn):
        return out_fn
    yadisk_key = 'https://yadi.sk/d/11eDCm7Dsn9GA'
    response = urlopen('https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
        + yadisk_key + '&path=/deep10M.fvecs')
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(',')[0][9:-1]
    filename = os.path.join(DATASET_DIR, 'deep-image.fvecs')
    download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]
    write_dense(out_fn, fv)
    return out_fn


# adapted from https://github.com/erikbern/ann-benchmarks.
def transform_bag_of_words(filename, n_dimensions, out_fn):
    import gzip
    from scipy.sparse import lil_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn import random_projection
    with gzip.open(filename, 'rb') as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(
            n_components=n_dimensions).fit_transform(B)
        print("Shape of C", C.shape)
        norms = np.linalg.norm(C, axis=1)
        print("Shape of norms", norms.shape)
        # remove entries with 0 norm
        D = np.delete(C, np.where(norms == 0.0)[0], axis=0).astype(np.float32)
        print("Shape of D", D.shape)
        write_dense(out_fn, D)
        return out_fn


# Adapted from https://github.com/erikbern/ann-benchmarks.
def nytimes(out_fn, n_dimensions):
    if os.path.isfile(out_fn):
        return out_fn
    fn = os.path.join(DATASET_DIR, 'nytimes_{}.txt.gz'.format(n_dimensions))
    download('https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz', fn)
    transform_bag_of_words(fn, n_dimensions, out_fn)
    return out_fn


# Adapted from https://github.com/erikbern/ann-benchmarks.
def kosarak(out_fn):
    if os.path.isfile(out_fn):
        return out_fn
    import gzip
    local_fn = 'kosarak.dat.gz'
    # only consider sets with at least min_elements many elements
    min_elements = 20
    url = 'http://fimi.uantwerpen.be/data/%s' % local_fn
    download(url, local_fn)

    X = []
    dimension = 0
    with gzip.open('kosarak.dat.gz', 'r') as f:
        content = f.readlines()
        # preprocess data to find sets with more than 20 elements
        # keep track of used ids for reenumeration
        for line in content:
            if len(line.split()) >= min_elements:
                X.append(list(map(int, line.split())))
                dimension = max(dimension, max(X[-1]) + 1)
    write_sparse(out_fn, X)
    return out_fn

# adapted from https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/datasets.py
def movielens(fn, ratings_file, out_fn, separator='::', ignore_header=False):
    import zipfile
    if os.path.isfile(out_fn):
        return out_fn

    url = 'http://files.grouplens.org/datasets/movielens/%s' % fn

    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        file = z.open(ratings_file)
        if ignore_header:
            file.readline()

        print('preparing %s' % out_fn)

        users = {}
        X = []
        dimension = 0
        for line in file:
            el = line.decode('UTF-8').split(separator)

            userId = el[0]
            itemId = int(el[1])
            rating = float(el[2])

            if rating < 3: # We only keep ratings >= 3
                continue

            if not userId in users:
                users[userId] = len(users)
                X.append([])

            X[users[userId]].append(itemId)
            dimension = max(dimension, itemId+1)

        write_sparse(out_fn, X)
    return out_fn

def movielens1m(out_fn):
    return movielens('ml-1m.zip', 'ml-1m/ratings.dat', out_fn)

def movielens10m(out_fn):
    return movielens('ml-10m.zip', 'ml-10M100K/ratings.dat', out_fn)

def movielens20m(out_fn):
    return movielens('ml-20m.zip', 'ml-20m/ratings.csv', out_fn, ',', True)


CREATE_RAW = os.path.abspath("join-experiments/scripts/createraw.py")
ORKUT_PY = os.path.abspath("join-experiments/scripts/orkut.py")


# Adapted from https://github.com/Cecca/danny/blob/master/datasets/prepare.py
def orkut(out_fn):
    if os.path.isfile(out_fn):
        return out_fn
    local_fn = os.path.join(DATASET_DIR, "orkut.txt.gz")
    directory = DATASET_DIR
    download("https://socialnetworks.mpi-sws.org/data/orkut-groupmemberships.txt.gz", local_fn)
    tmp = os.path.join(DATASET_DIR, "orkut.tmp")
    if not os.path.isfile(tmp):
        print("Building temporary file")
        subprocess.run(
            "cat {} | gunzip > tt.txt".format(local_fn), cwd=directory, shell=True, check=True
        )
        subprocess.run(
            "{} users tt.txt > orkut.out".format(ORKUT_PY), cwd=directory, shell=True, check=True
        )
        subprocess.run(
            "{} --bywhitespace orkut.out orkut-userswithgroups-raw.txt".format(CREATE_RAW),
            cwd=directory,
            shell=True,
            check=True
        )
        subprocess.run(
            "uniq orkut-userswithgroups-raw.txt > orkut.tmp", cwd=directory, shell=True, check=True
        )

    sets = []
    with open(tmp) as fp:
        for line in fp.readlines():
            tokens = line.split()
            items = [int(t) for t in tokens[2:]]
            sets.append(items)
    write_sparse(out_fn, sets)

    return out_fn


def dblp(out_fn):
    """Downloads and preprocesses a snapshot of DBLP"""
    import xml.sax
    from xml.sax.handler import ContentHandler

    class DblpFirstPassHandler(ContentHandler):
        def __init__(self, hdf_file):
            self.hdf_file = hdf_file
            self.tokens = dict() # Tokens with their count
            self.current_tag = None
            self.current_authors = []
            self.current_title = None
            self.progress = tqdm(unit = "papers")

        def startElement(self, tag, attrs):
            self.current_tag = tag
            return super().startElement(tag, attrs)

        def characters(self, content):
            if self.current_tag == "author":
                content = content.strip(" \n").lower()
                if len(content) > 0:
                    if not content in self.tokens:
                        self.tokens[content] = 1
                    else:
                        self.tokens[content] += 1
            elif self.current_tag == "title":
                self.progress.update(1)
                for tok in content.split():
                    tok = tok.strip(" \n").lower()
                    if len(tok) > 0:
                        if not tok in self.tokens:
                            self.tokens[tok] = 1
                        else:
                            self.tokens[tok] += 1
            return super().characters(content)

        def endElement(self, tag):
            self.current_tag = None
            return super().endElement(tag)

        def save_dictionary(self):
            """save the dictionary in the HDF5 file"""
            self.progress.close()
            print("Saving dictionary to hdf5 file")
            dictionary = sorted(self.tokens.items(), key=lambda pair: pair[1], reverse=False)
            words_dataset = self.hdf_file.create_dataset('dictionary/words', shape=(len(dictionary),), dtype=h5py.string_dtype())
            counts_dataset = self.hdf_file.create_dataset('dictionary/counts', shape=(len(dictionary),), dtype=np.int32)
            words_dataset[:] = [w for w, _ in dictionary]
            counts_dataset[:] = [c for _, c in dictionary]
            self.hdf_file.attrs['universe'] = len(dictionary)
            self.hdf_file.attrs['type'] = 'sparse'
            self.hdf_file.attrs['distance'] = 'jaccard'
            self.hdf_file.flush()

        def dictionary(self):
            """Returns all the words in decreasing order of frequency"""
            print("building dictionary")
            for tok, _c in sorted(self.tokens.items(), key=lambda pair: pair[1], reverse=False):
                yield tok


    class DblpSecondPassHandler(ContentHandler):
        def __init__(self, hdf_file):
            self.hdf_file = hdf_file
            self.current_tag = None
            self.current_vec = set()
            self.sets = set()
            self.dictionary = {}
            print("Reading dictionary from HDF5 file")
            for i, w in enumerate(self.hdf_file["dictionary/words"]):
                self.dictionary[w.decode('utf-8')] = i

            self.progress = tqdm(unit = "papers")

        def startElement(self, tag, attrs):
            self.current_tag = tag
            return super().startElement(tag, attrs)

        def characters(self, content):
            if self.current_tag == "author":
                content = content.strip(" \n").lower()
                if len(content) > 0:
                    self.current_vec.add(self.dictionary[content])
            elif self.current_tag == "title":
                self.progress.update(1)
                for tok in content.split():
                    if len(tok) > 0:
                        tok = tok.strip(" \n").lower()
                        self.current_vec.add(self.dictionary[tok])
            return super().characters(content)

        def endElement(self, tag):
            if tag == "article":
                res = tuple(sorted(self.current_vec))
                self.sets.add(res)
                self.current_vec = set()
            self.current_tag = None
            return super().endElement(tag)

        def save_sets(self):
            self.progress.close()
            sets = sorted(self.sets, key=lambda s: len(s))
            self.hdf_file.close()
            write_sparse(out_fn, sets)

    if os.path.isfile(out_fn):
        return out_fn
    url = "https://dblp.uni-trier.de/xml/dblp.xml.gz"
    local = os.path.join(DATASET_DIR, "dblp.xml.gz")
    download(url, local)
    hdf5_file = h5py.File(out_fn, "w")

    # First pass
    with gzip.open(local, 'rt', encoding='utf-8') as fp:
        handler = DblpFirstPassHandler(hdf5_file)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
        handler.save_dictionary()

    with gzip.open(local, 'rt', encoding='utf-8') as fp:
        handler = DblpSecondPassHandler(hdf5_file)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
        handler.save_sets()

    hdf5_file.close()
    return out_fn


# =============================================================================
# Putting it all together
# =======================
#
# Here we define some dictionaries mapping short names to algorithm configurations
# and datasets, to be used to run experiments

DATASETS = {
    'glove-25': lambda: glove(os.path.join(DATASET_DIR, 'glove-25.hdf5'), 25),
    'glove-200': lambda: glove(os.path.join(DATASET_DIR, 'glove-200.hdf5'), 200),
    'random-jaccard-10k': lambda: random_jaccard(os.path.join(DATASET_DIR, 'random-jaccard-10k.hdf5'), n=10000),
    'DBLP': lambda : dblp(os.path.join(DATASET_DIR, 'dblp.hdf5')),
    'Kosarak': lambda: kosarak(os.path.join(DATASET_DIR, 'kosarak.hdf5')),
    'DeepImage': lambda: deep_image(os.path.join(DATASET_DIR, 'deep_image.hdf5')),
    'NYTimes': lambda: nytimes(os.path.join(DATASET_DIR, 'nytimes.hdf5'), 256),
    'random-float-10k': lambda: random_float(os.path.join(DATASET_DIR, 'random-float-10k.hdf5' ), 20, 10000, 100),
    'random-difficult': lambda: random_difficult(os.path.join(DATASET_DIR, 'random-float-difficult.hdf5' ), 1100000, 150, 10),
    'Orkut': lambda: orkut(os.path.join(DATASET_DIR, "orkut.hdf5")),
    'movielens-1M': lambda: movielens1m(os.path.join(DATASET_DIR, "movielens-1M.hdf5")),
    'movielens-10M': lambda: movielens10m(os.path.join(DATASET_DIR, "movielens-10M.hdf5")),
    'movielens-20M': lambda: movielens20m(os.path.join(DATASET_DIR, "movielens-20M.hdf5")),
}

# Stores lazily the algorithm (i.e. as funcions to be called) along with their version
ALGORITHMS = {
    'PUFFINN':         lambda: (SubprocessAlgorithm(["build/PuffinnJoin"]), 6),
    # Local top-k baselines
    'BruteForceLocal': lambda: (BruteForceLocal(),                          1),
    'faiss-HNSW':      lambda: (FaissHNSW(),                                1),
    'faiss-IVF':       lambda: (FaissIVF(),                                 1),
    'pynndescent':     lambda: (PyNNDescent(),                              1),
    'falconn':         lambda: (FALCONN(),                                  2),
    # Global top-k baselines
    'XiaoEtAl':        lambda: (SubprocessAlgorithm(["build/XiaoEtAl"]),    1),
    'LSBTree':         lambda: (SubprocessAlgorithm(["build/LSBTree"]),     1)
}

# =============================================================================
# Utility functions
# =================

def get_output_file(configuration):
    """Returns a triplet: (filename, hdf5_path, hdf5_object)"""
    outdir = "output"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    fname = os.path.join(outdir, "{}-{}.hdf5".format(
        configuration['dataset'],
        configuration['algorithm']
    ))
    # Turns the dictionary of parameters into a string that
    # follows a consistent order
    params_list = sorted(list(configuration['params'].items()))
    params_string = ""
    for key, value in params_list:
        params_string += key.replace(" ", "") + str(value).replace(" ", "")
    params_hash = hashlib.sha256(params_string.encode('utf-8')).hexdigest()
    h5obj = h5py.File(os.path.join(BASE_DIR, fname), 'a')
    group = h5obj.require_group(params_hash)
    print(params_list)
    for key, value in params_list:
        print(key, value)
        group.attrs[key] = value
    return fname, params_hash, group


def run_config(configuration, debug=False):
    db = get_db()
    if not debug and already_run(db, configuration):
        print("Configuration already run, skipping")
        return
    output_file, group, output = get_output_file(configuration)
    hdf5_file = DATASETS[configuration['dataset']]()
    assert hdf5_file is not None
    algo, version = ALGORITHMS[configuration['algorithm']]()
    params = configuration['params']
    params['threads'] = configuration.get('threads', 1)
    k = configuration['k']
    if configuration['workload'] == 'local-top-k':
        hdf5_path = 'local-top-{}'.format(k)
    elif configuration['workload'] == 'global-top-k':
        hdf5_path = 'global-top-{}'.format(k)
    else:
        raise RuntimeError()
    print("=== k={} algorithm={} params={} dataset={}".format(
        k,
        configuration['algorithm'],
        params,
        configuration['dataset']
    ))
    time_index, time_workload = algo.execute(
        k,
        params,
        hdf5_file,
        output,
        hdf5_path
    )
    print("   time to index", time_index)
    print("   time for join", time_workload)
    if not debug:
        db.execute("""
        INSERT INTO main VALUES (
            :dataset,
            :workload,
            :k,
            :algorithm,
            :params,
            :threads,
            :time_index_s,
            :time_join_s,
            :recall,
            :output_file,
            :hdf5_group,
            :algorithm_version
        );
        """, {
            "dataset": configuration['dataset'],
            'workload': configuration['workload'],
            'k': k,
            'algorithm': configuration['algorithm'],
            'algorithm_version': version,
            'threads': configuration.get('threads', 1),
            'params': json.dumps(params, sort_keys=True),
            'time_index_s': time_index,
            'time_join_s': time_workload,
            'recall': None,
            'output_file': output_file,
            'hdf5_group': group
        })
    else:
        # Compute the recall and display it.
        baseline = get_baseline_indices(db, configuration['dataset'], k)
        with h5py.File(os.path.join(BASE_DIR, output_file), 'r') as hfp:
            actual_indices = np.array(hfp[group]['local-top-{}'.format(k)])
            avg_recall = compute_recall(k, baseline, actual_indices)
            print("Average recall", avg_recall)


def run_multiple(index_configuration, join_configurations, debug=False):
    """
    Instantiates an index with the given configuration, and run multiple joins, with different 
    parameterizations (as described in `join_configurations`) on the same index. 
    This allows to save the time of the indexing over multiple runs.
    """
    def merge_config(index_configuration, join_configuration):
        full_config = copy.deepcopy(index_configuration)
        full_config.update(index_configuration)
        full_config['params'].update(join_configuration)
        full_config['k'] = full_config['params']['k']
        del full_config['params']['k']
        return full_config

    db = get_db()

    hdf5_file = DATASETS[index_configuration['dataset']]()
    assert hdf5_file is not None
    algo, version = ALGORITHMS[index_configuration['algorithm']]()
    index_params = index_configuration['params']
    index_params['threads'] = index_configuration.get('threads', 1)

    print("BEFORE")
    pprint(index_configuration)
    configs_to_run = [
        join_config
        for join_config in join_configurations
        if not already_run(db, merge_config(
            copy.deepcopy(index_configuration), join_config
        ))
    ]
    print("AFTER")
    pprint(index_configuration)
    pprint(configs_to_run)

    if len(configs_to_run) == 0:
        return

    algo.feed_data(hdf5_file)
    print("Building index with parameters", index_params)
    try:
        algo.index(index_params)
    except:
        print("Error in index building")
        sys.exit(1)
        return

    for join_configuration in configs_to_run:
        algo.clear() # clears the result from the previous run

        full_config = merge_config(index_configuration, join_configuration)
        # full_config = index_configuration.copy()
        # full_config['params'].update(join_configuration)
        # full_config['k'] = full_config['params']['k']
        # del full_config['params']['k']
        # print("running with join configuration", json.dumps(full_config['params'], sort_keys=True))
        
        if already_run(db, full_config):
            print("Configuration already run, skipping")
            continue # skip this configuration
        print("running with full configuration", full_config)
        output_file, group, output = get_output_file(full_config)

        k = join_configuration['k']
        if full_config['workload'] == 'local-top-k':
            hdf5_path = 'local-top-{}'.format(k)
        elif full_config['workload'] == 'global-top-k':
            hdf5_path = 'global-top-{}'.format(k)
        else:
            raise RuntimeError("unsupported configuration")
        print("=== k={} algorithm={} params={} dataset={}".format(
            k,
            full_config['algorithm'],
            full_config['params'],
            full_config['dataset']
        ))

        algo.run(join_configuration)
        algo.save_result(output, hdf5_path)

        time_index, time_workload = algo.times() 

        print("   time to index", time_index)
        print("   time for join", time_workload)
        db.execute("""
        INSERT INTO main VALUES (
            :dataset,
            :workload,
            :k,
            :algorithm,
            :params,
            :threads,
            :time_index_s,
            :time_join_s,
            :recall,
            :output_file,
            :hdf5_group,
            :algorithm_version
        );
        """, {
            "dataset": full_config['dataset'],
            'workload': full_config['workload'],
            'k': k,
            'algorithm': full_config['algorithm'],
            'algorithm_version': version,
            'threads': full_config.get('threads', 1),
            'params': json.dumps(full_config['params'], sort_keys=True),
            'time_index_s': time_index,
            'time_join_s': time_workload,
            'recall': None,
            'output_file': output_file,
            'hdf5_group': group
        })
    algo.done()



if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        os.mkdir(BASE_DIR)
    
    for dummy_just_for_scoping in [0]:
        index_params = {
            'dataset': 'glove-200',
            'workload': 'local-top-k',
            'algorithm': 'BruteForceLocal',
            'params': {'prefix': 10000}
        } 
        query_params = [
            {'k': k}
            for k in [1000]
        ]

        run_multiple(index_params, query_params)
    
    with get_db() as db:
        compute_recalls(db)

    # run_config({
    #     'dataset': 'NYTimes',
    #     'workload': 'local-top-k',
    #     'k': 1000,
    #     'algorithm': 'BruteForceLocal',
    #     'params': {}
    # })

    # run_config({
    #     'dataset': 'DeepImage',
    #     'workload': 'local-top-k',
    #     'k': 1000,
    #     'algorithm': 'BruteForceLocal',
    #     'params': {'prefix': 10000}
    # })
    # run_config({
    #     'dataset': 'random-difficult',
    #     'workload': 'local-top-k',
    #     'k': 1000,
    #     'algorithm': 'BruteForceLocal',
    #     'params': {}
    # })

    threads = 56

    # for dataset in ['glove-200', 'DeepImage', 'DBLP', 'Orkut']:
    #     # ----------------------------------------------------------------------
    #     # Xiao et al. global top-k
    #     if dataset in ['DBLP', "Orkut"]:
    #         index_params = {
    #             'dataset': dataset,
    #             'workload': 'global-top-k',
    #             'algorithm': 'XiaoEtAl',
    #             'params': {}
    #         } 
    #         query_params = [
    #             {'k': k}
    #             for k in [1, 10, 100, 1000]
    #         ]
    #         run_multiple(index_params, query_params)

    #     # ----------------------------------------------------------------------
    #     # PUFFINN global top-k
    #     for hash_source in ['Independent']:
    #         space_usage = {
    #             'DeepImage': [32768, 65536],
    #             'glove-200': [2048, 4096, 8192, 16384],
    #             'Orkut': [16384, 32768],
    #             'DBLP': [2048, 4096, 8192, 16384],
    #         }
    #         for space_usage in space_usage[dataset]:
    #             index_params = {
    #                 'dataset': dataset,
    #                 'workload': 'global-top-k',
    #                 'algorithm': 'PUFFINN',
    #                 'threads': threads,
    #                 'params': {
    #                     'space_usage': space_usage,
    #                     'hash_source': hash_source
    #                 }
    #             }
    #             query_params = [
    #                 {'k': k, 'recall': recall, 'method': 'LSHJoinGlobal'}
    #                 for recall in [0.8, 0.9]
    #                 for k in [1, 10, 100, 1000]
    #             ]
    #             run_multiple(index_params, query_params)

    # ----------------------------------------------------------------------
    # LSB-Tree global top-k
    # for dataset in ['glove-25']:
    #     for m in [4, 8, 16]:
    #         for w in [1, 2, 4]:
    #             index_params = {
    #                 'dataset': dataset,
    #                 'workload': 'global-top-k',
    #                 'algorithm': 'LSBTree',
    #                 'params': {
    #                     'm': m,
    #                     'w': w
    #                 }
    #             }
    #             join_params = [
    #                 {'k': k}
    #                 for k in [10]
    #             ]
    #             run_multiple(index_params, join_params)

    for dataset in ['DBLP']:#, 'Orkut', 'DeepImage']:
        pass
        # ----------------------------------------------------------------------
        # pynndescent
        for n_neighbors in [20, 30, 50, 100]:
            for diversify_prob in [0.5, 0.75, 1.0]:
                for pruning_degree_multiplier in [1.0, 1.5]:
                    for leaf_size in [32]:
                        index_params = {
                            'n_neighbors': n_neighbors,
                            'leaf_size': leaf_size,
                            'pruning_degree_multiplier': pruning_degree_multiplier,
                            'diversify_prob': diversify_prob
                        }
                        join_params = [
                            {'k': k}
                            for k in [1, 10, 100]
                        ]
                        run_multiple(
                            {
                                'dataset': dataset,
                                'workload': 'local-top-k',
                                'algorithm': 'pynndescent',
                                'threads': threads,
                                'params': index_params
                            }, 
                            join_params
                        )

        # ----------------------------------------------------------------------
        # Faiss-HNSW
        # if dataset != 'DBLP':
        #     # for M in [4, 8, 16, 48, 32, 64, 128, 256, 512, 1024]:
        #     for M in [48, 32, 64]:
        #         for efConstruction in [100, 500]:
        #             index_params = {
        #                 'M': M,
        #                 'efConstruction': efConstruction
        #             }
        #             join_params = [
        #                 {'efSearch': efSearch, 'k': 10}
        #                 for efSearch in [80, 120]
        #                 # for efSearch in [10, 40, 80, 120, 800]
        #             ]
        #             run_multiple(
        #                 {
        #                     'dataset': dataset,
        #                     'workload': 'local-top-k',
        #                     'algorithm': 'faiss-HNSW',
        #                     'threads': threads,
        #                     'params': index_params
        #                 }, 
        #                 join_params
        #             )

        # ----------------------------------------------------------------------
        # Faiss-IVF
        # for n_list in [32, 64, 128, 256]:
        #     if dataset != 'DBLP':
        #         index_params = {
        #             'dataset': dataset,
        #             'workload': 'local-top-k',
        #             'algorithm': 'faiss-IVF',
        #             'threads': threads,
        #             'params': {
        #                 'n_list': n_list
        #             }
        #         }
        #         join_params = [
        #             {'n_probe': n_probe, 'k': 10}
        #             for n_probe in [1, 5, 10, 50]
        #         ]
        #         run_multiple(index_params, join_params)

        # ----------------------------------------------------------------------
        # FALCONN local top-k
        # for L in [5, 10, 50, 100]:
        #     if dataset != 'DBLP':
        #         index_params = {
        #             'dataset': dataset,
        #             'workload': 'local-top-k',
        #             'algorithm': 'falconn',
        #             'threads': threads,
        #             'params': {
        #                 "k": 3,
        #                 "L": L
        #             }
        #         }
        #         join_params = [
        #             {'k': 10, 'num_probes': num_probes, 'threads': threads}
        #             for num_probes in [L, 2 * L, 5 * L, 10 * L]
        #         ]
        #         run_multiple(index_params, join_params)

        # ----------------------------------------------------------------------
        # PUFFINN local top-k
        # for hash_source in ['Independent']:
        #     space_usage = {
        #         'DeepImage': [32768, 65536],
        #         'glove-200': [2048, 4096, 8192, 16384],
        #         'Orkut': [16384, 32768],
        #         'DBLP': [2048, 4096, 8192, 16384],
        #     }
        #     for space_usage in space_usage[dataset]:
        #         for sketches in ['1', '0']:
        #             index_params = {
        #                 'dataset': dataset,
        #                 'workload': 'local-top-k',
        #                 'algorithm': 'PUFFINN',
        #                 'threads': threads,
        #                 'params': {
        #                     'space_usage': space_usage,
        #                     'hash_source': hash_source,
        #                     'with_sketches': sketches
        #                 }
        #             }
        #             query_params = [
        #                 {'k': k, 'recall': recall, 'method': 'LSHJoin'}
        #                 for recall in [0.8, 0.9]
        #                 for k in [1, 10]
        #             ]
        #             run_multiple(index_params, query_params)




    with get_db() as db:
        compute_recalls(db)

