#!/usr/bin/env python3

# This script handles the execution of the join experiments 
# in two different modes:
#  - global top-k
#  - local top-k
# 
# Datasets are taken from ann-benchmarks or created ad-hoc from 
# other sources (e.g. DBLP).

from ast import Global
import numpy as np
import time
import subprocess
import h5py
import sys
import yaml
import shlex
import faiss
import os
import hashlib

# Communication protocol
# ======================
#
# For algorithms interfacing over text streams, the protocol is as follows:
#
# - Setup phase
#   - harness sends `sppv1 setup`, followed by parameters as `name value` pairs, followed by `sppv1 end`
#   - program acknowledges using `sppv1 ok`
# - Data ingestion
#   - harness sends `sppv1 data`, followed by vectors, one per line, followed by `sppv1 end`
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


def h5cat(path, dataset, stream=sys.stdout):
    file = h5py.File(path, "r")
    for v in file[dataset]:
        stream.write(text_encode_floats(v) + "\n")
        stream.flush()
        # print(text_encode_floats(v), file=stream)
    print(file=stream) # Signal end of streaming


class Algorithm(object):
    """Manages the lifecycle of an algorithm"""
    def execute(self, k, params, h5py_path, dataset, output_file, output_hdf5_path):
        self.setup(k, params)
        self.feed_data(h5py_path, dataset)
        self.index()
        self.run()
        self.save_result(output_file, output_hdf5_path)
        return self.times()
    def setup(self, k, params):
        """Configure the parameters of the algorithm"""
        pass
    def feed_data(self, h5py_path, dataset):
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
        hdf5_file[path] = self.result()
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
    def __init__(self, command_w_args):
        self.command_w_args = command_w_args
        self._program = None
        self.index_time = None
        self.workload_time = None

    def _subprocess_handle(self):
        if self._program is None:
            self._program = subprocess.Popen(
                self.command_w_args,
                bufsize=1,  # line buffering
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                universal_newlines=True)
        return self._program

    def setup(self, k, params_dict):
        print("Setup")
        self._send("setup")
        program = self._subprocess_handle()
        print("k", k, file=program.stdin)
        for key, v in params_dict.items():
            print(key, v, file=program.stdin)
        self._send("end")
        self._expect("ok", "setup failed")
        
    def feed_data(self, h5py_path, dataset):
        print("Feeding data using pipes")
        self._send("data")
        program = self._subprocess_handle()
        h5cat(h5py_path, dataset, program.stdin)
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
        return rows

    def times(self):
        return self.index_time, self.workload_time


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
    def feed_data(self, h5py_path, dataset):
        """Pass the data to the algorithm"""
        f = h5py.File(h5py_path)
        self.data = np.array(f[dataset])
        f.close()
    def index(self):
        """Setup the index, if any. This is timed."""
        print("  Building index")
        start = time.time()
        self.faiss_index = faiss.IndexHNSWFlat(len(self.data[0]), self.params["M"])
        self.faiss_index.hnsw.efConstruction = self.params["efConstruction"]
        self.faiss_index.add(self.data)
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
    



DATASETS = {
    'glove-25': ('/tmp/glove.hdf5', '/test')
}

ALGORITHMS = {
    'PUFFINN': SubprocessAlgorithm(["build/PuffinnJoin"]),
    'faiss-HNSW': FaissHNSW()
}


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
    h5obj = h5py.File(fname, 'a')
    group = h5obj.require_group(params_hash)
    for key, value in params_list:
        group.attrs[key] = value
    return fname, params_hash, group


def run_config(configuration):
    output_file, group, output = get_output_file(configuration)
    hdf5_file, datapath = DATASETS[configuration['dataset']]
    algo = ALGORITHMS[configuration['algorithm']]
    params = configuration['params']
    k = configuration['k']
    if configuration['mode'] == 'local-top-k':
        hdf5_path = 'local-top-{}'.format(k)
    elif configuration['mode'] == 'global-top-k':
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
        datapath,
        output,
        hdf5_path
    )
    print("   time to index", time_index)
    print("   time for join", time_workload)


if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        configs = yaml.load(fp, yaml.SafeLoader)
    for config in configs:
        run_config(config)
