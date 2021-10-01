import pdb
import requests
import sys
import urllib.parse
import numpy as np
from collections import OrderedDict
import argparse
import config_utils as cf


class EnsembleNER:
    def __init__(self):
        print("Ensemble NER started")
        self.servers  = cf.read_config()["NER_SERVERS"]
        self.rfp = open("log_results.txt","w")
        self.dfp = open("log_debug.txt","w")

    def dispatch_request(self,url):
        max_retries = 10
        attempts = 0
        while True:
            try:
                r = requests.get(url,timeout=1000)
                if (r.status_code == 200):
                    return r
            except:
                print("Request:", url, " failed. Retrying...")
            attempts += 1
            if (attempts >= max_retries):
                print("Request:", url, " failed")
                break
    
    def tag_sentence_service(self,text):
        ner_str = self.tag_sentence(text,self.rfp,self.dfp)
        return ner_str

    def tag_sentence(self,sent,rfp,dfp):
        responses = []
        for server in self.servers: 
            r = self.dispatch_request(server+str(sent))
            response = r.text
            #print(response)
            arr = response.split('\n')
            responses.append(arr)

        #TBD. Assumes to servers
        ret_str = "BIO SERVER(A100 server) | PHI SERVER(bert based cased)\n"
        for r1,r2 in zip(responses[0],responses[1]):
            #print(r1,r2)
            if (len(r1) <= 1):
                continue
            ret_str += r1 + "    |   " + r2 + '\n'
        ret_str += '\n'
        print(ret_str)
        return  ret_str + '\n' 

    def tag_se_in_sentence(self,sent,rfp,dfp):
        return 


def test_canned_sentences(obj):
    rfp = open("results.txt","w")
    dfp = open("debug.txt","w")
    obj.tag_sentence("Parkinsons:__entity__ father was a naval engineer",rfp,dfp)
    obj.tag_sentence("Fyodor:__entity__ Mikhailovich:__entity__ Dostoevsky:__entity__ was treated for Parkinsons:__entity__",rfp,dfp)
    obj.tag_sentence("Paul:__entity__ Erdős:__entity__ died at 83:__entity__",rfp,dfp)
    obj.tag_sentence("ajit:__entity__ rajasekharan:__entity__ is an engineer:__entity__",rfp,dfp)
    obj.tag_sentence("Imatinib:__entity__ mesylate:__entity__ is a drug and is used to treat nsclc:__entity__",rfp,dfp)
    obj.tag_sentence("1880s:__entity__ marked the beginning of Jazz:__entity__",rfp,dfp)
    obj.tag_sentence("He:__entity__ flew from New:__entity__ York:__entity__ to SFO:__entity__",rfp,dfp)
    obj.tag_sentence("Lionel:__entity__ Ritchie:__entity__ was popular in the 1980s:__entity__",rfp,dfp)
    obj.tag_sentence("Lionel:__entity__ Ritchie:__entity__ was popular in the late:__entity__ eighties:__entity__",rfp,dfp)
    obj.tag_sentence("John:__entity__ Doe:__entity__ flew from New:__entity__ York:__entity__ to Rio:__entity__ De:__entity__ Janiro:__entity__ via Miami:__entity__",rfp,dfp)
    obj.tag_sentence("He felt New:__entity__ York:__entity__ has a chance to win this year's:__entity__ competition",rfp,dfp)
    obj.tag_sentence("In humans mutations:__entity__ in Foxp2:__entity__ leads to verbal:__entity__ dyspraxia:__entity__",rfp,dfp)
    obj.tag_sentence("The recent spread of Corona:__entity__ virus:__entity__ flu:__entity__ from China:__entity__ to Italy:__entity__,Iran, South Korea and Japan has caused global concern",rfp,dfp)
    obj.tag_sentence("Hotel:__entity__ California:__entity__ topped the singles chart",rfp,dfp)
    obj.tag_sentence("Elon:__entity__ Musk:__entity__ said Telsa:__entity__ will open a manufacturing plant in Europe:__entity__",rfp,dfp)
    obj.tag_sentence("He flew from New York:__entity__ to SFO:__entity__",rfp,dfp)
    rfp.close()

def run_test(file_name,obj):
    rfp = open("results.txt","w")
    dfp = open("debug.txt","w")
    with open(file_name) as fp:
        count = 1
        for line in fp:
            if (len(line) > 1):
                print(str(count) + "] ",line,end='')
                obj.tag_sentence(line,rfp,dfp)
                count += 1
    rfp.close()
    dfp.close()


def tag_single_entity_in_sentence(file_name,obj):
    rfp = open("results.txt","w")
    dfp = open("debug.txt","w")
    sfp = open("se_results.txt","w")
    with open(file_name) as fp:
        count = 1
        for line in fp:
            if (len(line) > 1):
                print(str(count) + "] ",line,end='')
                entity_arr,span_arr,terms_arr,ner_str = obj.tag_se_in_sentence(line,rfp,dfp)
                #print("*******************:",terms_arr[span_arr.index(1)][WORD_POS].rstrip(":"),entity_arr[0])
                #sfp.write(terms_arr[span_arr.index(1)][WORD_POS].rstrip(":") + " " + entity_arr[0] + "\n")
                count += 1
                sfp.flush()
                pdb.set_trace()
    rfp.close()
    sfp.close()
    dfp.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble server',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="",help='Input file required for run options batch,single')
    parser.add_argument('-option', action="store", dest="option",default="canned",help='Valid options are canned,batch,single. canned - test few canned sentences used in medium artice. batch - tag sentences in input file. Entities to be tagged are determing used POS tagging to find noun phrases. specific - tag specific entities in input file. The tagged word or phrases needs to be of the form w1:__entity_ w2:__entity_ Example:Her hypophysitis:__entity__ secondary to ipilimumab was well managed with supplemental:__entity__ hormones:__entity__')
    results = parser.parse_args()

    obj = EnsembleNER()
    if (results.option == "canned"):
        test_canned_sentences(obj)
    elif (results.option == "batch"):
        if (len(results.input) == 0):
            print("Input file needs to be specified")
        else:
            run_test(results.input,obj)
            print("Tags and sentences are written in results.txt and debug.txt")
    elif (results.option == "specific"):
        if (len(results.input) == 0):
            print("Input file needs to be specified")
        else:
            tag_single_entity_in_sentence(results.input,obj)
            print("Tags and sentences are written in results.txt and debug.txt")
    else:
        print("Invalid argument:\n")
        parser.print_help()
