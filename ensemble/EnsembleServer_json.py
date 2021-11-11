# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import ensemble
import aggregate_server_json
import pdb
import json

LOG_FILE = "query_response_log.txt"

singleton = None
try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

class NerServer(ResponseHandler.ResponseHandler):
    def __init__(self):
        print("This is the constructor method.")
    def handler(self,write_obj = None):
        print("In derived class")
        global singleton
        if singleton is None:
            singleton = open(LOG_FILE,"a")
        if (write_obj is not None):
            param =write_obj.path[1:]
            print("Orig Arg = ",param)
            param = '/'.join(param.split('/')[1:])
            print("Json API param removed Arg = ",param)
            param = urllib.parse.unquote(param)
            #out = singleton.tag_sentence_service(param)
            out = aggregate_server_json.fetch_all(param)
            out = json.dumps(out,indent=5)
            print(out)
            print("Task complete. Writing out:",len(out))
            if (len(out) >= 1):
                write_obj.wfile.write(out.encode())
            else:
                write_obj.wfile.write("0".encode())
            singleton.write(out)
            singleton.flush()
            print("Write complete. Returning from handler")
            #write_obj.wfile.write("\nNF_EOS\n".encode())








def my_test():
    cl = NerServer()

    cl.handler()




#my_test()
