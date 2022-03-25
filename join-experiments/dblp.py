#!/usr/bin/env python3

# Download the raw dump from DBLP and transform it into a
# dataset amenable for Jaccard similarity based joins.

import xml
import numpy as np
import sys
import os
import urllib
import urllib.request
import gzip
import xml.sax
from xml.sax.handler import ContentHandler
import h5py

# download the file, if not availabe already
URL = "https://dblp.uni-trier.de/xml/dblp.xml.gz"
LOCAL = "dblp.xml.gz"
LOCAL = "dblp.head.xml.gz"

# if not os.path.isfile(LOCAL):
#     response = urllib.request.urlretrieve(URL, LOCAL)

class FirstPassHandler(ContentHandler):
    def __init__(self, hdf_file):
        self.hdf_file = hdf_file
        self.tokens = dict() # Tokens with their count
        self.current_tag = None
        self.current_authors = []
        self.current_title = None

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

class SecondPassHandler(ContentHandler):
    def __init__(self, hdf_file):
        self.hdf_file = hdf_file
        self.current_tag = None
        self.current_vec = set()
        self.sets = set()
        self.dictionary = {}
        for i, w in enumerate(self.hdf_file["dictionary/words"]):
            self.dictionary[w.decode('utf-8')] = i

    def startElement(self, tag, attrs):
        self.current_tag = tag
        return super().startElement(tag, attrs)

    def characters(self, content):
        if self.current_tag == "author":
            content = content.strip(" \n").lower()
            if len(content) > 0:
                self.current_vec.add(self.dictionary[content])
        elif self.current_tag == "title":
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

# First pass, to build the dictionary
def first_pass(xml_file, hdf5_file):
    with gzip.open(xml_file, 'rt', encoding='utf-8') as fp:
        handler = FirstPassHandler(hdf5_file)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
        handler.save_dictionary()

def second_pass(xml_file, hdf5_file):
    with gzip.open(xml_file, 'rt', encoding='utf-8') as fp:
        handler = SecondPassHandler(hdf5_file)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
        handler.save_sets()


hdf5_file = h5py.File("dblp.h5", "a")
print(hdf5_file.keys())
if "dictionary" not in list(hdf5_file.keys()):
    first_pass(LOCAL, hdf5_file)
else:
    print("Dictionary already in file, skipping")

second_pass(LOCAL, hdf5_file)

hdf5_file.close()

# DICTIONARY_FILE = "dblp.dict.txt"
# if not os.path.isfile(DICTIONARY_FILE):
#     first_pass(LOCAL, DICTIONARY_FILE)

# # Second pass
# dictionary = dict()
# with open(DICTIONARY_FILE, "r") as fp:
#     dictionary = dict(
#         (tok.strip(" \n\t"), i)
#         for (i, tok) in enumerate(fp.readlines())
#     )

# OUTPUT_FILE = "dblp.vecs.txt"

# with open(OUTPUT_FILE, "w") as ofp:
#     print(str(len(dictionary)+1), file=ofp)
#     with gzip.open(LOCAL, 'rt', encoding='utf-8') as fp:
#         handler = SecondPassHandler(dictionary, ofp)
#         parser = xml.sax.make_parser()
#         parser.setFeature(xml.sax.handler.feature_namespaces, 0)
#         parser.setContentHandler(handler)
#         parser.parse(fp)
