#!/usr/bin/python

import sys
import re

convtype = sys.argv[1]
filename = sys.argv[2]

aggregate = {}

pattern = None
if convtype == "users":
	pattern = re.compile(r'(?P<key>\d+)\s+(?P<val>\d+)\s*')
elif convtype == "groups":
	pattern = re.compile(r'(?P<val>\d+)\s+(?P<key>\d+)\s*')
	

with open(filename) as r:
	for line in r:
		m = pattern.match(line)
		if not m:
			raise Exception("Found line not conforming to expectations")
		agkey = int(m.group("key"))
		agval = m.group("val")
		glist = aggregate.get(agkey, [])
		glist.append(agval)
		aggregate[agkey] = glist


for agkey, agvals in aggregate.iteritems():
	print " ".join(agvals)

