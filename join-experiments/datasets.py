#!/usr/bin/python3

import xml
import numpy as np
import sys
import zipfile
import os
import urllib
import urllib.request
import gzip
import xml.sax
from xml.sax.handler import ContentHandler
import h5py
from tqdm import tqdm
import random


def download(url, dest):
    if not os.path.isfile(dest):
        print("Downloading {} to {}".format(url, dest))
        urllib.request.urlretrieve(url, dest)
        
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
        flattened = [
            x 
            for s in sets
            for x in s
        ]
        lengths = [len(s) for s in sets]
        offsets = np.zeros_like(lengths)
        offsets[1:] = np.cumsum(lengths)[:-1]
        items = self.hdf_file.create_dataset("vectors/items", shape=(len(flattened),), dtype=np.int32)
        items[:] = flattened
        lengths = self.hdf_file.create_dataset("vectors/lengths", shape=(len(lengths),), dtype=np.int32)
        lengths[:] = lengths
        offsets = self.hdf_file.create_dataset("vectors/offsets", shape=(len(offsets),), dtype=np.int32)
        offsets[:] = offsets
        self.hdf_file.flush()


def dblp(out_fn):
    if os.path.isfile(out_fn):
        return
    url = "https://dblp.uni-trier.de/xml/dblp.xml.gz"
    local = "datasets/dblp.xml.gz"
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


def glove(out_fn, dims):
    if os.path.isfile(out_fn):
        return
    url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    localzip = "datasets/glove.twitter.27B.zip"
    download(url, localzip)

    with zipfile.ZipFile(localzip) as zp:
        z_fn = 'glove.twitter.27B.%dd.txt' % dims
        vecs = np.array([
            np.array([float(t) for t in line.split()[1:]])
            for line in zp.open(z_fn)
        ])
        hfile = h5py.File(out_fn, "w")
        hfile.attrs["dimensions"] = len(vecs[0])
        hfile.attrs["type"] = "dense"
        hfile.attrs["distance"] = "cosine"
        hfile.create_dataset("vectors", shape=(len(vecs), len(vecs[0])), dtype=vecs[0].dtype, data=vecs, compression="gzip")
        hfile.close()
    return out_fn


# Adapted from ann-benchmarks
def random_jaccard(out_fn, n=10000, size=50, universe=80):
    if os.path.isfile(out_fn):
        return
    random.seed(1)
    f = h5py.File(out_fn, 'w')
    l = list(range(universe))
    # We call the set of sets `train` to be compatible with datasets from
    # ann-benchmarks
    train = []
    for i in range(n):
        train.append(random.sample(l, size))

    train = np.array(list(map(sorted, train)))
    flat_train = np.hstack(train.flatten())
    f.create_dataset('train', (len(flat_train),), dtype=flat_train.dtype)[:] = flat_train
    f.create_dataset('size_train', (len(train),), dtype='i')[:] = list(map(len, train))
    f.close()
    return out_fn


def iter_sparse(fn, data_path, size_path):
    f = h5py.File(fn, 'r')
    data = np.array(f[data_path])
    sizes = np.array(f[size_path])
    offsets = np.zeros(sizes.shape, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    for start, s in zip(offsets, sizes):
        v = data[start:start+s]
        assert len(v) == s
        yield v


if not os.path.isdir("datasets"):
    os.mkdir("datasets")

DATASETS = {
    "DBLP": lambda: dblp("datasets/dblp.h5"),
    "Glove-25": lambda: glove("datasets/glove-25.h5", 25),
    "Random-Jaccard-10k": lambda: random_jaccard('datasets/random-jaccard.hdf5')
}

if __name__ == "__main__":
    # DATASETS["DBLP"]()
    # DATASETS["Glove-25"]()
    DATASETS["Random-Jaccard-10k"](),
    for v in iter_sparse(
        'datasets/random-jaccard.hdf5',
        'train',
        'size_train'
    ):
        print(v)
