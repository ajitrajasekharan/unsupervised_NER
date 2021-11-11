#!/usr/bin/python3
import threading
import time
import sys
import pdb
import requests
import urllib
from utils.common import *
import config_utils as cf


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

phi_precedence_arr = [
"PERSON",
"ORGANIZATION",
"ENT",
"LOCATION",
"COLOR",
"LANGUAGE",
"GRAMMAR_CONSTRUCT",
"OTHER",
"UNTAGGED_ENTITY",
"SOCIAL_CIRCUMSTANCES",
"MEASURE"
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
      self.raw_results = []
      self.results = []
   def run(self):
      print ("Starting " + self.url + self.param)
      out = requests.get(self.url + self.param)
      self.raw_results = out.text.split('\n')
      for line in self.raw_results:
          self.results.append(line)
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
    count = 1
    raw_results = []
    for thread in threads_arr:
        results.append("\n" +str(count) + "] " +  thread.desc + "\n\n" + '\n'.join(thread.results) + "\n\n")
        count += 1
        raw_results.append(thread.raw_results)
    return results,raw_results

#TBD. This needs to be done in a principled way.
#for this first cut, INCARCERATION in second server, created a prediction with the main entity type of server1 and serve2 combined
def override_bio_prediction1(orig_entity,frag,pos_index,servers,server_index):
    main_entity = frag.split()[pos_index].split('[')[0]
    if (main_entity == "SOCIAL_CIRCUMSTANCES" or main_entity == "ORGANIZATION" ):  #for the first cut
        return True
    return False

#TBD. This needs to be done in a principled way.
#for this first cut, measure in bio space, is overriden by specific entities in phi server
def override_bio_prediction2(orig_entity,frag,pos_index,servers,server_index):
    main_entity = frag.split()[pos_index].split('[')[0]
    if (orig_entity == "MEASURE"  and  main_entity in servers[server_index]["precedence"]):  #for the first cut, MEASURE in bio server can be overriden by PHI server if the corresponding entity prediction by PHI server is part of the PHI server list
        return True
    return False


def get_ensembled_entity_frags(entity_frags,servers_arr,entities_count):
    ensembled = []
    print("Ensemble candidates")
    print('\n'.join(entity_frags))
    print()
    for pos_index  in range(entities_count):
        match_arr = []
        found = False
        server_index = 0
        for frag in entity_frags:
            main_entity = frag.split()[pos_index].split('[')[0]
            if (main_entity in servers_arr[server_index]["precedence"]):
                    if (server_index == 0 and override_bio_prediction1(main_entity,entity_frags[server_index + 1 ],pos_index,servers_arr,server_index + 1)):
                        match_arr.append(entity_frags[server_index + 1 ].split()[pos_index].split('[')[0]  + '/' + main_entity)
                    elif (server_index == 0 and override_bio_prediction2(main_entity,entity_frags[server_index + 1 ],pos_index,servers_arr,server_index + 1)):
                        match_arr.append(entity_frags[server_index+1].split()[pos_index])
                    else:
                        match_arr.append(frag.split()[pos_index])
                    found = True
                    break
            if (found):
                break
            server_index += 1
        assert(len(match_arr) == 1)
        ensembled.append(match_arr[0])
    return ensembled



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


def ensemble_processing(sent,raw_results,results):
    terms_arr = set_POS_based_on_entities(sent)
    masked_sent_arr,span_arr = generate_masked_sentences(terms_arr)
    tag_count = len(masked_sent_arr)
    entity_frags = []
    count = 0
    for arr in raw_results:
        j = len(arr) - 1
        while (j >= 0):
            if (arr[j].startswith(RESULT_MASK)):
                result_str = arr[j].split(RESULT_MASK)[1].strip()
                entity_frags.append(result_str)
                assert(len(result_str.split()) == tag_count)
                count += 1
                break
            j -= 1
    assert(len(entity_frags) == len(actions_arr))
    ensembled_entity_frags = get_ensembled_entity_frags(entity_frags,actions_arr,tag_count)
    ret_str = gen_ensembled_sentence(sent,terms_arr,ensembled_entity_frags,span_arr)
    print(ret_str)
    results.insert(0,ret_str)

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
    results,raw_results = get_results(threads_arr)

    #this updates results with ensembled results
    ensemble_processing(inp,raw_results,results)
    if (query_log_fp is None):
        query_log_fp = open("query_logs.txt","a")
    query_log_fp.write(urllib.parse.unquote(inp)+"\n")
    query_log_fp.flush()
    end = time.time()
    results.insert(0,"\n\nEnsemble server count:" + str(len(actions_arr)) + ". (*** Scroll down to see second server result ***)\n")
    results.insert(1,"Elapsed time:" + str(round(end-start,1)) + " secs\n\n")
    print("Elapsed time:" + str(round(end-start,1)) + " secs\n\n")
    return results

def main():
    while True:
        print("Enter text with entity for masked position")
        inp = input()
        if (inp == "q" or inp  == "quit"):
            break
        results = fetch_all(inp)
        print ("Exiting Main Thread:",'\n'.join(results))

if __name__ == "__main__":
    main()
