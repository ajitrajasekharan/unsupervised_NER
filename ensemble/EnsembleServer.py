# -*- coding: utf-8 -*-
import os
import ResponseHandler
import subprocess
import urllib
import ensemble
import pdb


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
            singleton = ensemble.EnsembleNER()
        if (write_obj is not None):
            param =write_obj.path[1:]
            print("Orig Arg = ",param)
            param = '/'.join(param.split('/')[1:])
            print("API param removed Arg = ",param)
            param = urllib.parse.unquote(param)
            out = singleton.tag_sentence_service(param)
            #print("Arg = ",write_obj.path[1:])
            #out = singleton.punct_sentence(urllib.parse.unquote(write_obj.path[1:].lower()))
            print(out)
            if (len(out) >= 1):
                write_obj.wfile.write(out.encode())
            else:
                write_obj.wfile.write("0".encode())
            #write_obj.wfile.write("\nNF_EOS\n".encode())








def my_test():
    cl = NerServer()

    cl.handler()




#my_test()
