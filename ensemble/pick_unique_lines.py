import pdb
import config_utils as cf
import requests
import sys
import urllib.parse
import numpy as np
from collections import OrderedDict
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pick unuique files from log ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="query_logs.txt",help='Input file ')
    parser.add_argument('-output', action="store", dest="output",default="uniqified_query_logs.txt",help='Output file')
    results = parser.parse_args()

    unique_dict = {}
    total = 0
    with open(results.input) as fp:
        for line in fp:
            total += 1
            line = line.strip()
            if line not in unique_dict:
                unique_dict[line] = 1
    with open(results.output,"a") as fp:
        for key in unique_dict:
            fp.write(key + "\n")
    print("Original input line count",total)
    print("Unique lines output and appended to existing file:",len(unique_dict),results.output)


