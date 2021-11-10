#!/usr/bin/python3
import threading
import time
import sys
import pdb
import requests
import urllib
from utils.common import *
import config_utils as cf
import json
from  collections import OrderedDict
import argparse


MASK = ":__entity__"
RESULT_MASK = "NER_FINAL_RESULTS:"

bio_precedence_arr = [ "THERAPEUTIC_OR_PREVENTIVE_PROCEDURE",
"DISEASE",
"GENE",
"BODY_PART_OR_ORGAN_OR_ORGAN_COMPONENT",
"ORGANISM_FUNCTION",
"BIO",
"OBJECT",
"MEASURE"
]

#this is a catchall
phi_precedence_arr = [
"PERSON",
"ORGANIZATION",
"ENT",
"LOCATION",
"COLOR",
"LANGUAGE",
"GRAMMAR_CONSTRUCT",
"OTHER",
"SOCIAL_CIRCUMSTANCES",
"MEASURE",
"OBJECT",
"THERAPEUTIC_OR_PREVENTIVE_PROCEDURE",
"DISEASE",
"GENE",
"BODY_PART_OR_ORGAN_OR_ORGAN_COMPONENT",
"ORGANISM_FUNCTION",
"BIO",
"OBJECT",
"O"
]

actions_arr = [
        {"url":"http://127.0.0.1:8088/dummy/","desc":"****************** A100 trained Bio model (Pubmed,Clincial trials, Bookcorpus(subset) **********","precedence":bio_precedence_arr},
        {"url":"http://127.0.0.1:9088/dummy/","desc":"********** Bert base cased (bookcorpus and Wikipedia) ***********","precedence":phi_precedence_arr}
        ]

class myThread (threading.Thread):
   def __init__(self, url,param,desc):
      threading.Thread.__init__(self)
      self.url = url
      self.param = param
      self.desc = desc
      self.results = {}
   def run(self):
      print ("Starting " + self.url + self.param)
      out = requests.get(self.url + self.param)
      self.results = json.loads(out.text,object_pairs_hook=OrderedDict)
      self.results["server"] = self.desc
      print ("Exiting " + self.url + self.param)


# Create new threads
def create_workers(inp_dict,inp):
    threads_arr = []
    for i in range(len(inp_dict)):
        threads_arr.append(myThread(inp_dict[i]["url"],inp,inp_dict[i]["desc"]))
    return threads_arr

def start_workers(threads_arr):
    for thread in threads_arr:
        thread.start()

def wait_for_completion(threads_arr):
    for thread in threads_arr:
        thread.join()

def get_results(threads_arr):
    results = []
    for thread in threads_arr:
        results.append(thread.results)
    return results

#TBD. This needs to be done in a principled way.
#for this first cut, INCARCERATION in second server, created a prediction with the main entity type of server1 and serve2 combined
def override_bio_prediction1(orig_entity,frag,pos_index,servers,server_index):
    main_entity = prefix_strip(frag.split('[')[0])
    if (main_entity == "SOCIAL_CIRCUMSTANCES" or main_entity == "ORGANIZATION" ):  #for the first cut
        return True
    return False

#TBD. This needs to be done in a principled way.
#for this first cut, measure in bio space, is overriden by specific entities in phi server
def override_bio_prediction2(orig_entity,frag,pos_index,servers,server_index):
    main_entity = prefix_strip(frag.split('[')[0])
    if (orig_entity == "MEASURE"  and  main_entity in servers[server_index]["precedence"]):  #for the first cut, MEASURE in bio server can be overriden by PHI server if the corresponding entity prediction by PHI server is part of the PHI server list
        return True
    return False


def confirm_same_size_responses(results):
    count = 0
    for i in range(len(results)):
        ner = results[i]["ner"]
        if(count == 0):
            assert(len(ner) > 0)
            count = len(ner)
        else: 
            if (count != len(ner)):
                assert(0)
    return count

def prefix_strip(term):
    if (term.startswith("B_") or term.startswith("I_")):
        term = term[2:]
    return term

def get_ensembled_entities(results,servers_arr):
    ensembled = OrderedDict()
    ensembled_conf =  OrderedDict()
    print("Ensemble candidates")
    terms_count =  confirm_same_size_responses(results)
    assert(len(servers_arr) == len(results))
    for term_index  in range(terms_count):
        match_dict = {}
        conf_dict = {}
        found = False
        pos_index = str(term_index + 1)
        for server_index in range(len(results)):
            main_entity = results[server_index]["ner"][pos_index]["e"].split('[')[0] 
            main_entity = prefix_strip(main_entity)
            if (main_entity in servers_arr[server_index]["precedence"]):
                    if (server_index == 0 and override_bio_prediction1(main_entity,results[server_index + 1 ]["ner"][pos_index]["e"],pos_index,servers_arr,server_index + 1)):
                        match_dict[pos_index]  = results[server_index + 1 ]["ner"][pos_index]["e"].split('[')[0]  + '/' + main_entity
                        if (pos_index in  results[server_index + 1]["cs_prediction_details"]):
                            conf_dict[pos_index] = results[server_index + 1]["cs_prediction_details"][pos_index]
                    elif (server_index == 0 and override_bio_prediction2(main_entity,results[server_index + 1 ]["ner"][pos_index]["e"],pos_index,servers_arr,server_index + 1)):
                        match_dict[pos_index] = results[server_index+1]["ner"][pos_index]
                        if (pos_index in  results[server_index + 1]["cs_prediction_details"]):
                            conf_dict[pos_index] = results[server_index + 1]["cs_prediction_details"][pos_index]
                    else:
                        match_dict[pos_index] = results[server_index]["ner"][pos_index]
                        if (pos_index in  results[server_index]["cs_prediction_details"]):
                            conf_dict[pos_index] = results[server_index]["cs_prediction_details"][pos_index]
                    found = True
                    break
            if (found):
                break
        if (len(match_dict) != 1):
            pdb.set_trace()
        assert(len(match_dict) == 1)
        first_key = next(iter(match_dict))
        assert(first_key not in ensembled)
        ensembled[first_key]  = match_dict[first_key]
        if (len(conf_dict) > 0):
            assert(len(conf_dict) == 1)
            first_key = next(iter(conf_dict))
            assert(first_key not in ensembled_conf)
            ensembled_conf[first_key]  = conf_dict[first_key]
    return ensembled,ensembled_conf


def gen_ensembled_sentence(sent,terms_arr,detected_entities_arr,span_arr):
    #print("Final result")
    ret_str = ""
    #for i,term in enumerate(terms_arr):
    #    print(term[WORD_POS],' ',end='')
    #print()
    sent_arr = sent.split()
    assert(len(terms_arr) == len(span_arr))
    entity_index = 0
    i = 0
    in_span = False
    while (i < len(span_arr)):
        if (span_arr[i] == 0):
            tag = "O"
            if (in_span):
                in_span = False
                entity_index += 1
        else:
            if (in_span):
                tag = "I_" + detected_entities_arr[entity_index]
            else:
                in_span = True
                tag = "B_" + detected_entities_arr[entity_index]
        ret_str = ret_str + terms_arr[i][WORD_POS] + ' ' + tag + "\n"
        #print(tag + ' ',end='')
        i += 1
    #print()
    ret_str += "\n"
    return ret_str


def ensemble_processing(sent,results):
    ensembled_ner,ensembled_conf = get_ensembled_entities(results,actions_arr)
    final_ner = OrderedDict()
    final_ner["ensembled_ner"] = ensembled_ner
    final_ner["ensembled_prediction_details"] = ensembled_conf
    final_ner["individual"] = results
    return final_ner


query_log_fp = None

def fetch_all(inp):
    global query_log_fp
    start = time.time()
    print("Starting threads")
    servers  = cf.read_config()["NER_SERVERS"]
    if (len(servers) > 0):
        assert(len(actions_arr) == len(servers)) #currently just using two servers. TBD. Fix
        for i in range(len(actions_arr)):
            actions_arr[i]["url"] = servers[i]
    threads_arr = create_workers(actions_arr,inp)
    start_workers(threads_arr)
    wait_for_completion(threads_arr)
    print("All threads complete")
    results = get_results(threads_arr)
    #print(json.dumps(results,indent=4))

    #this updates results with ensembled results
    results = ensemble_processing(inp,results)
    if (query_log_fp is None):
        query_log_fp = open("query_logs.txt","a")
    query_log_fp.write(urllib.parse.unquote(inp)+"\n")
    query_log_fp.flush()
    end = time.time()
    results["stats"] = { "Ensemble server count" : str(len(actions_arr)),
                         "Elapsed time" : str(round(end-start,1)) + " secs"}
    return results

def tag_interactive():
    while True:
        print("Enter text with entity for masked position")
        inp = input()
        if (inp == "q" or inp  == "quit"):
            break
        results = fetch_all(inp)
        print(json.dumps(results,indent=4))

def run_test(inp_file):
    with open(inp_file) as fp:
        for line in fp:
            line = line.rstrip('\n')
            results = fetch_all(line)
            print("Input:",line)
            print(json.dumps(results,indent=4))
            pdb.set_trace()


canned_sentences = [
    "After studies at Hofstra University , He worked for New York Telephone before He was elected to the New York State Assembly to represent the 16th District in Northwest Nassau County ",
    "Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",
    "Imatinib mesylate is a drug and is used to treat nsclc",
    "engineer",
    "Austin called",
    "Her hypophysitis secondary to ipilimumab was well managed with supplemental hormones",
    "ajit rajasekharan is an engineer",
    "Paul Erdős died at 83",
    "In Seattle , Pete Incaviglia 's grand slam with one out in the sixth snapped a tie and lifted the Baltimore Orioles past the Seattle           Mariners , 5-2 .",
    "It was Incaviglia 's sixth grand slam and 200th homer of his career .",
    "Add Women 's singles , third round Lisa Raymond ( U.S. , beat Kimberly Po ( U.S. , 6-3 6-2 .",
    "1880s marked the beginning of Jazz",
    "He flew from New York to SFO",
    "Lionel Ritchie was popular in the 1980s",
    "Lionel Ritchie was popular in the late eighties",
    "John Doe flew from New York to Rio De Janiro via Miami",
    "He felt New York has a chance to win this year's competition",
    "Fyodor Mikhailovich Dostoevsky was treated for Parkinsons",
    "In humans mutations in Foxp2 leads to verbal dyspraxia",
    "The recent spread of Corona virus flu from China to Italy,Iran, South Korea and Japan has caused global concern",
    "Hotel California topped the singles chart",
    "Elon Musk said Telsa will open a manufacturing plant in Europe",
    "He flew from New York to SFO",
    "Everyday he rode his bicycle from Rajakilpakkam to Tambaram",
    "If he loses Saturday , it could devalue his position as one of the world 's great boxers , \" Panamanian Boxing Association President Ramon     Manzanares said .",
    "West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",
    "they are his friends ",
    "they flew from Boston to Rio De Janiro and had a mocha",
    "he flew from Boston to Rio De Janiro and had a mocha",
    "X,Y,Z are medicines"
]

def test_canned_sentences():
    for line in canned_sentences:
        results = fetch_all(line)
        print("Input:",line)
        print(json.dumps(results,indent=4))
        #pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main NER for a single model ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="",help='Input file required for run options batch,single')
    parser.add_argument('-option', action="store", dest="option",default="canned",help='Valid options are canned,batch,interactive. canned - test few canned sentences used in medium artice. batch - tag sentences in input file. Entities to be tagged are determing used POS tagging to find noun phrases.interactive - input one sentence at a time')
    results = parser.parse_args()

    if (results.option == "interactive"):
        tag_interactive()
    elif (results.option == "batch"):
        if (len(results.input) == 0):
            print("Input file needs to be specified")
        else:
            run_test(results.input)
    else:
        test_canned_sentences()
