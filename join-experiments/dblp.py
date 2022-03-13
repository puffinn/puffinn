#!/usr/bin/env python3

# Download the raw dump from DBLP and transform it into a
# dataset amenable for Jaccard similarity based joins.

import xml
import sys
import os
import urllib
import urllib.request
import gzip
import xml.sax
from xml.sax.handler import ContentHandler

# download the file, if not availabe already
URL = "https://dblp.uni-trier.de/xml/dblp.xml.gz"
LOCAL = "dblp.xml.gz"

if not os.path.isfile(LOCAL):
    response = urllib.request.urlretrieve(URL, LOCAL)

class FirstPassHandler(ContentHandler):
    def __init__(self):
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
        return super().endElement(tag)

    def dictionary(self):
        """Returns all the words in decreasing order of frequency"""
        print("building dictionary")
        for tok, _c in sorted(self.tokens.items(), key=lambda pair: pair[1], reverse=False):
            yield tok

class SecondPassHandler(ContentHandler):
    def __init__(self, dictionary, file_output):
        self.dictionary = dictionary
        self.file_output = file_output
        self.current_tag = None
        self.current_vec = set()
        self.already_emitted = set()

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
            to_emit = " ".join(str(i) for i in sorted(self.current_vec))
            if to_emit not in self.already_emitted:
                self.file_output.write(to_emit)
                self.file_output.write("\n")
                self.current_vec = set()
                self.already_emitted.add(to_emit)
        return super().endElement(tag)

# First pass, to build the dictionary
def first_pass(xml_file, dictionary_file):
    with gzip.open(xml_file, 'rt', encoding='utf-8') as fp:
        handler = FirstPassHandler()
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
    with open(dictionary_file, "w") as fp:
        # Retrieve tokens by frequency
        for token in handler.dictionary():
            print(token, file=fp)

DICTIONARY_FILE = "dblp.dict.txt"
if not os.path.isfile(DICTIONARY_FILE):
    first_pass(LOCAL, DICTIONARY_FILE)

# Second pass
dictionary = dict()
with open(DICTIONARY_FILE, "r") as fp:
    dictionary = dict(
        (tok.strip(" \n\t"), i)
        for (i, tok) in enumerate(fp.readlines())
    )

OUTPUT_FILE = "dblp.vecs.txt"

with open(OUTPUT_FILE, "w") as ofp:
    print(str(len(dictionary)+1), file=ofp)
    with gzip.open(LOCAL, 'rt', encoding='utf-8') as fp:
        handler = SecondPassHandler(dictionary, ofp)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        parser.setContentHandler(handler)
        parser.parse(fp)
