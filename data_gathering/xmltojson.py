# Given a path (-p) to a directory of annotation xml files, this program will write them out as jsons

import json
from optparse import OptionParser
import re
import os
import xmltodict as xd
from collections import OrderedDict
from itertools import combinations

parser = OptionParser()

parser.add_option("-t", "--top", dest="top",
                  help="Path to top-level dir of xml files.")
parser.add_option("-p", "--path", dest="path",
                  help="Path to dir of xml files.")
parser.add_option("-o", "--output_path", dest="output_path",
                  help="Output path")

(options, args) = parser.parse_args()
if not options.path:
    if not options.top:
    	parser.error(
	       'Error: path to test data must be specified. Pass --top to command line')
    else:
        options.path = os.path.join(options.top, 'xml_annotations')

if not options.output_path:   # if filename is not given
    if not options.top:
        parser.error(
           'Error: path to test data must be specified. Pass --top to command line')
    else:
        options.output_path = os.path.join(options.top, 'annotations')

if options.output_path:
    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)


xmls = sorted(os.listdir(options.path))

for x in xmls:
    if not x.endswith('.xml'):
        continue
    with open(os.path.join(options.path, x), 'r+') as f:
        x_imageDict = xd.parse(f)
        f.close()

    print('File: {}'.format(x))
    with open(os.path.join(options.output_path, x.split('.')[0] + '.json'), 'w') as fp:
        json.dump(x_imageDict, fp)
        fp.close()
