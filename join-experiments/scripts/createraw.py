#!/usr/bin/python

import argparse
import re
from collections import deque
from itertools import chain
import codecs

def tokenize_qgram(line, q):
	tokens = []
	curtoken = deque([ '#' ] * q)


	for s in chain(line, ["#"] * (q - 1)):
		curtoken.append(s)
		curtoken.popleft()
		tokens.append("".join([ codecs.encode(ct, "utf8") for ct in curtoken ]))

	return tokens

def tokenize_whitespace(line, filterre=r'\s+'):
	tokens = re.split(filterre, line)

	# remove last token if it is empty...
	if tokens[-1] == '':
		del tokens[-1]

	return tokens

def tokenize_shingles(line, shilen, filterre=r'\s+'):
	tokens = []
	curtoken = deque([ '#' ] * shilen)

	linesplit = tokenize_whitespace(line, filterre)

	for s in chain(linesplit, ["#"] * (shilen - 1)):
		curtoken.append(s)
		curtoken.popleft()
		tokens.append(tuple(curtoken))

	return tokens


def sort_len_lex(x, y):
	if len(x) != len(y):
		return cmp(len(x), len(y))
	return cmp(x, y)
#	for i in range(len(x)):
#		if x[i] != y[i]:
#			return cmp(x[i], y[i])
#	return cmp(True, True)


parser = argparse.ArgumentParser()
inputtreatmentgroup = parser.add_mutually_exclusive_group(required=True)
inputtreatmentgroup.add_argument("--qgram", type=int)
inputtreatmentgroup.add_argument("--separator", type=str)
inputtreatmentgroup.add_argument("--shingles", type=int)
inputtreatmentgroup.add_argument("--bywhitespace", action="store_true")
parser.add_argument("--foreign", type=str)
parser.add_argument("--foreign-output", type=str)
parser.add_argument("--alphanum", action="store_true", help="Separate by non-alphanumeric characters (only relevant to --bywhitespace and --shingles options")
parser.add_argument("--uppercase", action="store_true", help="Convert strings to upper case")
parser.add_argument("--dedup", action="store_true", help="Deduplicate sets")
parser.add_argument("--dedupitems", action="store_true", help="Deduplicate set items")
parser.add_argument("indexed")
parser.add_argument("indexedoutput")

args = parser.parse_args()

filterre = r'\s*'
if args.alphanum:
	filterre = r'\W+'

if args.qgram:
	tokenizer = lambda x : tokenize_qgram(x, args.qgram)
elif args.shingles:
	tokenizer = lambda x : tokenize_shingles(x, args.shingles, filterre)
elif args.separator:
	tokenizer = lambda x : tokenize_whitespace(x, args.separator)
else:
	tokenizer = lambda x : tokenize_whitespace(x, filterre)

indexedrecords = []
foreignrecords = []
tokenmap = {}

nexttokenid = 1

with codecs.open(args.indexed, encoding='utf8') as indf:
	for line in indf:
		if args.uppercase:
			line = line.upper()
		record = []
		rectokens = {}
		tokens = tokenizer(line.rstrip('\n'))

		if len(tokens) == 0:
			continue

		for token in tokens:
			# get number of times token occurred so far in record
			tokcnt = rectokens.get(token, 0)

			# If token has already occurred on line and dedupitems given, ignore it
			if args.dedupitems and tokcnt != 0:
				continue

			# assign key for global tokenmap
			tokenmapkey = (token,tokcnt)
			# update the local index token count
			rectokens[token] = tokcnt + 1
			# if tokenkey already seen
			if tokenmapkey in tokenmap:
				# append to record
				record.append(tokenmap[tokenmapkey])
			# otherwise
			else:
				# store tokenid in global tokenmap
				tokenmap[tokenmapkey] = nexttokenid
				# append to record
				record.append(nexttokenid)
				# update nexttokenid
				nexttokenid += 1
		indexedrecords.append(record)

if args.foreign:
	with codecs.open(args.foreign, encoding='utf8') as forf:
		for line in forf:
			record = []
			rectokens = {}
			tokens = tokenizer(line.rstrip('\n'))

			if len(tokens) == 0:
				continue

			for token in tokens:
				# get number of times token occurred so far in record
				tokcnt = rectokens.get(token, 0)
				# assign key for global tokenmap
				tokenmapkey = (token,tokcnt)
				# update the local index token count
				rectokens[token] = tokcnt + 1
				# if tokenkey already seen
				if tokenmapkey in tokenmap:
					# append to record
					record.append(tokenmap[tokenmapkey])
				# otherwise
				else:
					# append to record
					record.append(0)
			foreignrecords.append(record)
				
ntokens = len(tokenmap)
print("There are {} tokens".format(len(tokenmap)))
tokenmap.clear()

tokencountlist = [ [0, i]  for i in xrange(nexttokenid) ]
for record in indexedrecords:
	for token in record:
		tokencountlist[token][0] += 1

tokencountlist.sort(cmp=lambda x,y: cmp(x[0],y[0]))
tokenmaplist = [0] * len(tokencountlist)

for i in xrange(len(tokencountlist)):
	tokenmaplist[tokencountlist[i][1]] = i

for record in indexedrecords:
	for i in range(len(record)):
		record[i] = tokenmaplist[record[i]]
	record.sort()

for record in foreignrecords:
	for i in range(len(record)):
		record[i] = tokenmaplist[record[i]]
	record.sort()

indexedrecords.sort(cmp=sort_len_lex)
foreignrecords.sort(cmp=sort_len_lex)

lastrecord = []

with open(args.indexedoutput, "w") as indout:
	cnt = 0
	for record in indexedrecords:
		if args.dedup and lastrecord == record:
			continue
		print >> indout, str(cnt) + "  " + str(ntokens) + " " + " ".join([ str(nmb) for nmb in record ])
		lastrecord = record
		cnt += 1

lastrecord = []

if args.foreign:
	with open(args.foreign_output, "w") as forout:
		cnt = 0
		for record in foreignrecords:
			if args.dedup and lastrecord == record:
				continue
			print >> forout, str(cnt) + " " + str(ntokens) + " " + " ".join([ str(nmb) for nmb in record ])
			lastrecord = record
