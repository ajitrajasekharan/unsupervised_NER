import torch
import pdb
import operator
from collections import OrderedDict
import sys
import traceback
import argparse
import string


UTAG = "UNTAGGED_ENTITY"



def merge_entities(final_dict,file_name):
    picked = 0
    with open(file_name) as fp:
        for line in fp:
            line = line.rstrip('\n').split()
            if (len(line) == 2):
                key = line[0].lower() #all terms are saved a lower case
                if (key not in final_dict):
                    final_dict[key] = []
                entities = line[1].rstrip('/').split('/')
                for entity in entities:
                    if (entity not in final_dict[key]):
                        final_dict[key].append(entity)
                picked += 1
    return picked




def construct(params):
    files_list = params.list
    output_file =  params.output
    final_dict = OrderedDict()
    total_picked = 0
    with open(files_list) as fp:
        for line in fp:
            line = line.rstrip('\n')
            print("File:",line)
            total_picked += merge_entities(final_dict,line)

    print("total entities",total_picked)
    wfp = open(output_file,"w")
    for term in final_dict:

        if (UTAG in final_dict[term]):
            if (len(final_dict[term]) > 1):
                oterm = ' '.join(final_dict[term]).replace(UTAG,'').split()
            else:
                oterm = final_dict[term]
        else:
            oterm = final_dict[term]

        wfp.write('/'.join(oterm)+' '+term+'\n')
    wfp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge all entities list into one bootstrap list ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-list', action="store", dest="list", default="list.txt",help='list of entities files')
    parser.add_argument('-output', action="store", dest="output", default="bootstrap_entities.txt",help='list of entities files')
    results = parser.parse_args()
    try:
        construct(results)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
