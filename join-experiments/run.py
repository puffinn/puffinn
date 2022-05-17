#!/usr/bin/env python3

# This script handles the execution of the join experiments
# in two different modes:
#  - global top-k
#  - local top-k
#
# Datasets are taken from ann-benchmarks or created ad-hoc from
# other sources (e.g. DBLP).

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
import faiss
import os
import hashlib
import sqlite3
import json
import random
import numba
from urllib.request import urlopen
from urllib.request import urlretrieve

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
    configuration['threads'] = configuration.get('threads', 1)
    configuration['params']['threads'] = configuration.get('threads', 1)
    configuration['params'] = json.dumps(configuration['params'], sort_keys=True)
    res = db.execute("""
    SELECT rowid FROM main
    WHERE dataset = :dataset
      AND workload = :workload
      AND threads = :threads
      AND k = :k
      AND algorithm = :algorithm
      AND params = :params
    """, configuration).fetchall()
    return len(res) > 0


def compute_recalls(db):
    # Local topk
    missing_recalls = db.execute("SELECT rowid, algorithm, params, dataset, k, output_file, hdf5_group FROM main WHERE recall IS NULL AND WORKLOAD = 'local-top-k';").fetchall()
    for rowid, algorithm, params, dataset, k, output_file, hdf5_group in missing_recalls:
        print("Computing recalls for {} {} on {} with k={}".format(algorithm, params, dataset, k))
        baseline = db.execute(
            "SELECT output_file, hdf5_group FROM baselines WHERE dataset = :dataset AND workload = 'local-top-k';",
            {"dataset": dataset}
        ).fetchone()
        if baseline is None:
            print("Missing baseline")
            continue
        base_file, base_group = baseline
        base_file = os.path.join(BASE_DIR, base_file)
        output_file = os.path.join(BASE_DIR, output_file)
        print("output file", output_file, "rowid", rowid, "group", hdf5_group)
        with h5py.File(base_file) as hfp:
            baseline_indices = hfp[base_group]['local-top-1000'][:,:k]
        with h5py.File(output_file) as hfp:
            actual_indices = np.array(hfp[hdf5_group]['local-top-{}'.format(k)])
        assert baseline_indices.shape == actual_indices.shape
        recalls = np.array([
            np.mean(np.isin(baseline_indices[i], actual_indices[i]))
            for i in tqdm(range(len(baseline_indices)), leave=False)
        ])
        with h5py.File(output_file, 'r+') as hfp:
            hfp[hdf5_group]['local-top-{}-recalls'.format(k)] = recalls
        avg_recall = np.mean(recalls)
        db.execute(
            """UPDATE main
                 SET recall = :recall
               WHERE rowid = :rowid ;
            """,
            {"rowid": rowid, "recall": avg_recall}
        )
    # Global topk
    missing_recalls = db.execute("SELECT rowid, algorithm, params, dataset, k, output_file, hdf5_group FROM main WHERE recall IS NULL AND WORKLOAD = 'global-top-k';").fetchall()
    for rowid, algorithm, params, dataset, k, output_file, hdf5_group in missing_recalls:
        print("Computing recalls for {} {} on {} with k={}".format(algorithm, params, dataset, k))
        baseline = db.execute(
            "SELECT output_file, hdf5_group FROM baselines_global WHERE dataset = :dataset AND k = :k AND workload = 'global-top-k';",
            {"dataset": dataset, "k": k}
        ).fetchone()
        if baseline is None:
            print("Missing baseline")
            continue
        print(baseline)
        base_file, base_group = baseline
        base_file = os.path.join(BASE_DIR, base_file)
        output_file = os.path.join(BASE_DIR, output_file)
        with h5py.File(base_file) as hfp:
            baseline_pairs = set(map(tuple, hfp[base_group]['global-top-{}'.format(k)]))
        with h5py.File(output_file) as hfp:
            actual_pairs = set(map(tuple, hfp[hdf5_group]['global-top-{}'.format(k)]))
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
# - Setup phase
#   - harness sends `sppv1 setup`, followed by parameters as `name value` pairs, followed by `sppv1 end`
#   - program acknowledges using `sppv1 ok`
# - Data ingestion
#   - harness sends `sppv1 data`, `sspv1 distance_type`, followed by vectors, one per line, followed by `sppv1 end`
#   - program acknowledges using `sppv1 ok`
# - Index construction (timed)
#   - harness sends `sppv1 index`
#   - program acknowledges using `sppv1 ok` when it's done
# - Workload run (timed)
#   - harness sends `sppv1 workload`
#   - program acknowledges using `sppv1 ok` when it's done
# - Result collection
#   - harness sends `sppv1 result`
#   - program writes output on stdout, one item per line (for whatever "item" means for the workload)
#   - program writes `sppv1 end`

PROTOCOL = "sppv1" # Version of the communication protocol


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
        self.setup(k, params)
        self.feed_data(h5py_path)
        self.index()
        self.run()
        self.save_result(output_file, output_hdf5_path)
        return self.times()
    def setup(self, k, params):
        """Configure the parameters of the algorithm"""
        pass
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        pass
    def index(self):
        """Setup the index, if any. This is timed."""
        pass
    def run(self):
        """Run the workload. This is timed."""
        pass
    def result(self):
        """Return the result as a two-dimensional numpy array.
        The way it is interpreted depends on the application.
        For global top-k join, it is the list of top-k pairs of indices.
        For local top-k join, it is the list of
        nearest neighbors for each element.
        """
        pass
    def save_result(self, hdf5_file, path):
        result = self.result()
        if path in hdf5_file:
            existing = np.array(hdf5_file[path])
            print(existing)
            assert (existing == result).all()
            return
        hdf5_file[path] = result
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        pass


class SubprocessAlgorithm(Algorithm):
    """Manages the lifecycle of an algorithm which does not provide
    a Python interface"""

    def _raw_line(self):
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

    def setup(self, k, params_dict):
        print("Setup")
        self._send("setup")
        program = self._subprocess_handle()
        print("k", k, file=program.stdin)
        for key, v in params_dict.items():
            print(key, v, file=program.stdin)
        self._send("end")
        self._expect("ok", "setup failed")

    def feed_data(self, h5py_path):
        distance = h5py.File(h5py_path).attrs['distance']
        self._send("data")
        self._send(distance)
        program = self._subprocess_handle()
        self._send("path " + h5py_path)
        self._expect("ok", "population phase failed")

    def index(self):
        print("Building index")
        start_t = time.time()
        self._send("index")
        self._expect("ok", "workload phase failed")
        end_t = time.time()
        self.index_time = end_t - start_t

    def run(self):
        print("Running workload")
        start_t = time.time()
        self._send("workload")
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
            if line[1] == "end":
                break
            rows.append(np.array([int(i) for i in line]))
        rows = np.array(rows)
        print("Waiting for subprocess to finish")
        self._wait_for_completion()
        return rows

    def times(self):
        return self.index_time, self.workload_time


class FaissIVF(Algorithm):
    def __init__(self):
        self.faiss_index = None
        self.k = None
        self.params = None
        self.data = None
        self.time_index = None
        self.time_run = None
        self.result_indices = None
    def setup(self, k, params):
        """Configure the parameters of the algorithm"""
        self.k = k
        self.params = params
        faiss.omp_set_num_threads(params['threads'])
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train'])
        f.close()
    def index(self):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()

        n_list = self.params["n_list"]
        n_probe = self.params["n_probe"]

        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], n_list, faiss.METRIC_L2)
        index.nprobe = n_probe
        index.train(X)
        index.add(X)
        self.faiss_index = index
        self.time_index = time.time() - start
    def run(self):
        """Run the workload. This is timed."""
        print("  Top-{} join".format(self.k))
        start = time.time()
        _dists, idxs = self.faiss_index.search(self.data, self.k+1)
        self.time_run = time.time() - start
        self.result_indices = idxs[:,1:]
    def result(self):
        return self.result_indices
    def times(self):
        """Returns the pair (index_time, workload_time)"""
        return self.time_index, self.time_run



class FaissHNSW(Algorithm):
    def __init__(self):
        self.faiss_index = None
        self.k = None
        self.params = None
        self.data = None
        self.time_index = None
        self.time_run = None
        self.result_indices = None
    def setup(self, k, params):
        """Configure the parameters of the algorithm"""
        self.k = k
        self.params = params
        faiss.omp_set_num_threads(params['threads'])
    def feed_data(self, h5py_path):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train'])
        f.close()
    def index(self):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()
        X = sklearn.preprocessing.normalize(self.data, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.faiss_index = faiss.IndexHNSWFlat(len(X[0]), self.params["M"])
        self.faiss_index.hnsw.efConstruction = self.params["efConstruction"]
        self.faiss_index.add(X)
        self.time_index = time.time() - start
    def run(self):
        """Run the workload. This is timed."""
        print("  Top-{} join".format(self.k))
        self.faiss_index.hnsw.efSearch = self.params["efSearch"]
        start = time.time()
        _dists, idxs = self.faiss_index.search(self.data, self.k+1)
        self.time_run = time.time() - start
        self.result_indices = idxs[:,1:]
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
    def setup(self, k, params):
        self.k = k
        pass
    def feed_data(self, h5py_path):
        f = h5py.File(h5py_path)
        assert f.attrs['distance'] == 'cosine' or f.attrs['distance'] == 'angular'
        self.data = np.array(f['train'])
        norms = np.linalg.norm(self.data, axis=1)[:, np.newaxis]
        assert np.sum(norms == 0.0) == 0
        self.data /= norms
        f.close()
    def index(self):
        self.time_index = 0
    def run(self):
        print("  Top-{} join".format(self.k))
        start = time.time()
        index = faiss.IndexFlatIP(len(self.data[0]))
        index.add(self.data)
        self.result_indices = index.search(self.data, self.k+1)[1][:, 1:]
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

# Adapted from ann-benchmarks
def random_jaccard(out_fn, n=10000, size=50, universe=80):
    if os.path.isfile(out_fn):
        return
    random.seed(1)
    l = list(range(universe))
    # We call the set of sets `train` to be compatible with datasets from
    # ann-benchmarks
    train = []
    for i in range(n):
        train.append(random.sample(l, size))

    write_sparse(out_fn, train)
    return out_fn


def glove(out_fn, dims):
    if os.path.isfile(out_fn):
        return out_fn
    url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    localzip = "datasets/glove.twitter.27B.zip"
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
    'random-jaccard-10k': lambda: random_jaccard(os.path.join(DATASET_DIR, 'random-jaccard-10k.hdf5'), n=10000),
    'DBLP': lambda : dblp(os.path.join(DATASET_DIR, 'dblp.hdf5')),
    'Kosarak': lambda: kosarak(os.path.join(DATASET_DIR, 'kosarak.hdf5')),
    'DeepImage': lambda: deep_image(os.path.join(DATASET_DIR, 'deep_image.hdf5')),
    'NYTimes': lambda: nytimes(os.path.join(DATASET_DIR, 'nytimes.hdf5'), 256),
}

ALGORITHMS = {
    'PUFFINN': lambda: SubprocessAlgorithm(["build/PuffinnJoin"]),
    # Local top-k baselines
    'BruteForceLocal': lambda: BruteForceLocal(),
    'faiss-HNSW': lambda: FaissHNSW(),
    'faiss-IVF': lambda: FaissIVF(),
    # Global top-k baselines
    'XiaoEtAl': lambda: SubprocessAlgorithm(["build/XiaoEtAl"])
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
    for key, value in params_list:
        group.attrs[key] = value
    return fname, params_hash, group


def run_config(configuration):
    db = get_db()
    if already_run(db, configuration):
        print("Configuration already run, skipping")
        return
    output_file, group, output = get_output_file(configuration)
    hdf5_file = DATASETS[configuration['dataset']]()
    assert hdf5_file is not None
    algo = ALGORITHMS[configuration['algorithm']]()
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
        :hdf5_group
    );
    """, {
        "dataset": configuration['dataset'],
        'workload': configuration['workload'],
        'k': k,
        'algorithm': configuration['algorithm'],
        'threads': configuration.get('threads', 1),
        'params': json.dumps(params, sort_keys=True),
        'time_index_s': time_index,
        'time_join_s': time_workload,
        'recall': None,
        'output_file': output_file,
        'hdf5_group': group
    })


if __name__ == "__main__":
    if not os.path.isdir(BASE_DIR):
        os.mkdir(BASE_DIR)

    with get_db() as db:
        compute_recalls(db)

    run_config({
        'dataset': 'NYTimes',
        'workload': 'local-top-k',
        'k': 1000,
        'algorithm': 'BruteForceLocal',
        'params': {}
    })

    # run_config({
    #     'dataset': 'glove-25',
    #     'workload': 'local-top-k',
    #     'k': 1000,
    #     'algorithm': 'BruteForceLocal',
    #     'params': {}
    # })

    threads = 56

    # ----------------------------------------------------------------------
    # Xiao et al. global top-k
    # for k in [1, 10, 100, 1000]:
    #     run_config({
    #         'dataset': 'DBLP',
    #         'workload': 'global-top-k',
    #         'k': k,
    #         'algorithm': 'XiaoEtAl',
    #         'params': {}
    #     })

    for dataset in ['NYTimes', 'glove-25']:
        # ----------------------------------------------------------------------
        # Faiss-HNSW
        # for M in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for M in [48, 64]:
            for efConstruction in [100, 500]:
                for efSearch in [10, 40, 80, 120, 800]:
                    run_config({
                        'dataset': dataset,
                        'workload': 'local-top-k',
                        'k': 10,
                        'algorithm': 'faiss-HNSW',
                        'threads': threads,
                        'params': {
                            'M': M,
                            'efConstruction': efConstruction,
                            'efSearch': efSearch,
                        }
                    })

        # ----------------------------------------------------------------------
        # Faiss-IVF
        for n_list in [32, 64, 128, 256]:
            for n_probe in [1, 5, 10, 50]:
                run_config({
                    'dataset': dataset,
                    'workload': 'local-top-k',
                    'k': 10,
                    'algorithm': 'faiss-IVF',
                    'threads': threads,
                    'params': {
                        'n_list': n_list,
                        'n_probe': n_probe
                    }
                })

        # ----------------------------------------------------------------------
        # PUFFINN local top-k
        for hash_source in ['Independent', 'Tensored']:
            for recall in [0.8, 0.9, 0.99]:
                for space_usage in [1024, 2048, 4096]:
                    run_config({
                        'dataset': dataset,
                        'workload': 'local-top-k',
                        'k': 10,
                        'algorithm': 'PUFFINN',
                        'threads': threads,
                        'params': {
                            'method': 'LSHJoin',
                            'recall': recall,
                            'space_usage': space_usage,
                            'hash_source': hash_source
                        }
                    })

    # run_config({
    #     'dataset': 'glove-25',
    #     'workload': 'local-top-k',
    #     'k': 5,
    #     'algorithm': 'PUFFINN',
    #     'threads': threads,
    #     'params': {
    #         'method': 'LSHJoin',
    #         'recall': 0.9,
    #         'space_usage': 1024
    #     }
    # })

    with get_db() as db:
        compute_recalls(db)

