# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import batched_main_NER
import pdb
import config_utils as cf


DEFAULT_CONFIG = "./config.json"

singleton = None
full_sentence_tag = True
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class BatchedNerServer(ResponseHandler.ResponseHandler):
    def __init__(self):
        print("This is the constructor method.")
    def handler(self,write_obj = None):
        print("In derived class")
        global singleton
        global full_sentence_tag
        if singleton is None:
            singleton = batched_main_NER.UnsupNER(DEFAULT_CONFIG)
            full_sentence_tag  = True if cf.read_config()["FULL_SENTENCE_TAG"] == "1" else False
        if (write_obj is not None):
            param =write_obj.path[1:]
            print("Orig Arg = ",param)
            param = '/'.join(param.split('/')[1:])
            print("API param removed Arg = ",param)
            param = urllib.parse.unquote(param)
            out = singleton.tag_sentence_service(param,full_sentence_tag)
            #print("Arg = ",write_obj.path[1:])
            #out = singleton.punct_sentence(urllib.parse.unquote(write_obj.path[1:].lower()))
            print(out)
            if (len(out) >= 1):
                write_obj.wfile.write(out.encode())
            else:
                write_obj.wfile.write("0".encode())
            #write_obj.wfile.write("\nNF_EOS\n".encode())








def my_test():
    cl = EntityFilter()

    cl.handler()




#my_test()
