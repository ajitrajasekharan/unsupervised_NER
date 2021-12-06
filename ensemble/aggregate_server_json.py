#!/usr/bin/python3
import threading
import time
import math
import sys
import pdb
import requests
import urllib
from utils.common import *
import config_utils as cf
import json
from  collections import OrderedDict
import argparse
import numpy as np
import aggregate_server_json


MASK = ":__entity__"
RESULT_MASK = "NER_FINAL_RESULTS:"

DEFAULT_TEST_BATCH_FILE="bootstrap_test_set.txt"
NER_OUTPUT_FILE="ner_output.txt"
DEFUALT_THRESHOLD = 1 #1 standard deviation from nean - for cross over prediction

actions_arr = [
        {"url":cf.read_config()["actions_arr"][0]["url"],"desc":cf.read_config()["actions_arr"][0]["desc"], "precedence":cf.read_config()["bio_precedence_arr"],"common":cf.read_config()["common_entities_arr"]},
        {"url":cf.read_config()["actions_arr"][1]["url"],"desc":cf.read_config()["actions_arr"][1]["desc"],"precedence":cf.read_config()["phi_precedence_arr"],"common":cf.read_config()["common_entities_arr"]},
        ]

class AggregateNER:
    def __init__(self):
        self.rfp = open("query_response_log.txt","a")
        self.query_log_fp = open("query_logs.txt","a")
        self.inferred_entities_log_fp = open("inferred_entities_log.txt","a")
        self.threshold = DEFUALT_THRESHOLD #TBD read this from confg. cf.read_config()["CROSS_OVER_THRESHOLD_SIGMA"]
        self.servers  = cf.read_config()["NER_SERVERS"]


    def fetch_all(self,inp):
        start = time.time()
        self.query_log_fp.write(urllib.parse.unquote(inp)+"\n")
        self.query_log_fp.flush()
        print("Starting threads")
        if (len(self.servers) > 0):
         assert(len(actions_arr) == len(self.servers)) #currently just using two servers. TBD. Fix
         for i in range(len(actions_arr)):
             actions_arr[i]["url"] = self.servers[i]
        threads_arr = create_workers(actions_arr,inp)
        start_workers(threads_arr)
        wait_for_completion(threads_arr)
        print("All threads complete")
        results = get_results(threads_arr)
        #print(json.dumps(results,indent=4))

        #this updates results with ensembled results
        results = self.ensemble_processing(inp,results)
        end = time.time()

        results["stats"] = { "Ensemble server count" : str(len(actions_arr)),
                          "Elapsed time" : str(round(end-start,1)) + " secs"}

        self.rfp.write( "\n" + json.dumps(results,indent=4))
        self.rfp.flush()
        return results


    def get_conflict_resolved_entity(self,results,term_index,terms_count,servers_arr):
        pos_index = str(term_index + 1)
        s1_entity  = extract_main_entity(results,0,pos_index)
        s2_entity  = extract_main_entity(results,1,pos_index)
        span_count1 = get_span_info(results,0,term_index,terms_count)
        span_count2 = get_span_info(results,1,term_index,terms_count)
        assert(span_count1 == span_count2)
        if (s1_entity == s2_entity):
            server_index = 0 if (s1_entity in servers_arr[0]["precedence"]) else 1
            if (s1_entity != "O"):
                print("Both servers agree on prediction for term:",results[0]["ner"][pos_index]["term"],":",s1_entity)
            return server_index,span_count1,-1
        else:
            print("Servers do not agree on prediction for term:",results[0]["ner"][pos_index]["term"],":",s1_entity,s2_entity)
            #Both the servers dont agree on their predictions. First server is BIO server. Second is PHI
            #Examine both server predictions.
            #Case 1: If one of them cross predicts, dump it.
                #Return the other prediction (TBD. Assert the cross prediction is indeed part of the other servers above mean prediction?
                #if not treat again as a tie and return both?).
                #If both predict anyone of the common enitites "e.g. MEASURE", then treat it is as case 2. Both cross predict
            #Case 2. If both cross predict, then return both predictions - this is a case both servers have low confidence
            #Case 3: Both dont cross predict. Then just return both predictions with BIO prediction listed first.
            #Cross prediction is checked only for  predictions a server makes ABOVE prediction  mean.
            cross_predictions,cross_prediction_count = self.check_cross_server_prediction(results,term_index,servers_arr)
            if (cross_prediction_count == 1):
                ret_server_index = 1 if cross_predictions[0] == True else 0
            else:
                ret_server_index = 0
        return ret_server_index,span_count1,cross_prediction_count

    def check_cross_server_prediction(self,results,term_index,servers_arr):
        pos_index = str(term_index + 1)
        cross_predictions = {}
        cross_prediction_count = 0
        for server_index in range(len(results)):
            if (pos_index in  results[server_index]["entity_distribution"]):
                 predictions = self.get_predictions_above_threshold(results[server_index]["entity_distribution"][pos_index])
                 is_included = is_included_in_server_entities(predictions,servers_arr[server_index])
                 cross_predictions[server_index] = not is_included
                 cross_prediction_count += 1 if not is_included else 0
        print("Cross prediction count for term: " + results[0]["ner"][pos_index]["term"] + " is :: " + str(cross_prediction_count))
        return cross_predictions,cross_prediction_count

    def  get_predictions_above_threshold(self,predictions):
        dist = predictions["cs_distribution"]
        sum_predictions = 0
        ret_arr = []
        assert(len(dist) != 0)
        mean_score = 1.0/len(dist) #input is a prob distriubution. so sum is 1
        sum_deviation = 0
        for node in dist:
            sum_deviation += (mean_score - node["confidence"])*(mean_score - node["confidence"])
        variance = sum_deviation/len(dist)
        std_dev = math.sqrt(variance)
        threshold =  mean_score + std_dev*self.threshold #default is 1 standard deviation from mean
        for node in dist:
            if (node["confidence"] > threshold):
                ret_arr.append(node["e"])
            else:
                break #this is a reverse sorted list. So no need to check anymore
        return ret_arr

    def gen_resolved_entity(self,results,server_index,run_index,cross_prediction_count):
        if (cross_prediction_count == 1 or cross_prediction_count == -1):
            #return results[server_index]["ner"][run_index]
            return flip_category(results[server_index]["ner"][run_index])
        else:
            ret_obj = results[server_index]["ner"][run_index].copy()
            #ret_obj["e"] = results[0]["ner"][run_index]["e"] + "/" + results[1]["ner"][run_index]["e"]
            n1 = flip_category(results[0]["ner"][run_index])
            n2 = flip_category(results[1]["ner"][run_index])
            ret_obj["e"] = n1["e"] + "/" + n2["e"]
            return ret_obj




    def get_ensembled_entities(self,results,servers_arr):
        ensembled_ner = OrderedDict()
        ensembled_conf =  OrderedDict()
        ambig_ensembled_conf =  OrderedDict()
        ensembled_ci = OrderedDict()
        ensembled_cs = OrderedDict()
        ambig_ensembled_ci = OrderedDict()
        ambig_ensembled_cs = OrderedDict()
        print("Ensemble candidates")
        terms_count =  confirm_same_size_responses(results)
        if (terms_count == 0):
            return ensembled_ner,ensembled_conf,ensembled_ci,ensembled_cs,ambig_ensembled_conf,ambig_ensembled_ci,ambig_ensembled_cs
        assert(len(servers_arr) == len(results))
        term_index = 0
        while (term_index  < terms_count):
            pos_index = str(term_index + 1)
            assert(len(servers_arr) == 2) #TBD. Currently assumes two servers in prototype to see if this approach works. To be extended to multiple servers
            server_index,span_count,cross_prediction_count = self.get_conflict_resolved_entity(results,term_index,terms_count,servers_arr)
            for span_index in range(span_count):
                run_index = str(term_index + 1 + span_index)
                ensembled_ner[run_index] = self.gen_resolved_entity(results,server_index,run_index,cross_prediction_count)
                if (run_index in  results[server_index]["entity_distribution"]):
                    ensembled_conf[run_index] = results[server_index]["entity_distribution"][run_index]
                    ensembled_conf[run_index]["e"] = ensembled_ner[run_index]["e"] #this is to make sure the same tag can be taken from NER result or this structure.
                                                                                   #When both server responses are required, just return the details of first server for now
                    ensembled_ci[run_index] = results[server_index]["ci_prediction_details"][run_index]
                    ensembled_cs[run_index] = results[server_index]["cs_prediction_details"][run_index]

                    if (cross_prediction_count == 0 or cross_prediction_count == 2): #This is an ambiguous prediction. Send both server responses
                        second_server = server_index + 1
                        ambig_ensembled_conf[run_index] = results[second_server]["entity_distribution"][run_index]
                        ambig_ensembled_conf[run_index]["e"] = ensembled_ner[run_index]["e"] #this is to make sure the same tag can be taken from NER result or this structure.
                        ambig_ensembled_ci[run_index] = results[second_server]["ci_prediction_details"][run_index]
                        ambig_ensembled_cs[run_index] = results[second_server]["cs_prediction_details"][run_index]
                if (ensembled_ner[run_index]["e"] != "O"):
                    self.inferred_entities_log_fp.write(results[0]["ner"][run_index]["term"] + " " + ensembled_ner[run_index]["e"]  + "\n")
            term_index += span_count
        self.inferred_entities_log_fp.flush()
        return ensembled_ner,ensembled_conf,ensembled_ci,ensembled_cs,ambig_ensembled_conf,ambig_ensembled_ci,ambig_ensembled_cs



    def ensemble_processing(self,sent,results):
        ensembled_ner,ensembled_conf,ci_details,cs_details,ambig_ensembled_conf,ambig_ci_details,ambig_cs_details = self.get_ensembled_entities(results,actions_arr)
        final_ner = OrderedDict()
        final_ner["ensembled_ner"] = ensembled_ner
        final_ner["ensembled_prediction_details"] = ensembled_conf
        final_ner["ci_prediction_details"] = ci_details
        final_ner["cs_prediction_details"] = cs_details
        final_ner["ambig_prediction_details_conf"] = ambig_ensembled_conf
        final_ner["ambig_prediction_details_ci"] = ambig_ci_details
        final_ner["ambig_prediction_details_cs"] = ambig_cs_details
        #final_ner["individual"] = results
        return final_ner




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


def confirm_same_size_responses(results):
    count = 0
    for i in range(len(results)):
        if ("ner" in results[i]):
            ner = results[i]["ner"]
        else:
            print("Server",i," returned invalid response;",results[i])
            return 0
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



#This hack is simply done for downstream API used for UI displays the entity instead of the class. Details has all additional info
def flip_category(obj):
    new_obj = obj.copy()
    entity_type_arr = obj["e"].split("[")
    if (len(entity_type_arr) > 1):
        term = entity_type_arr[0]
        if (term.startswith("B_") or term.startswith("I_")):
            prefix = term[:2]
            new_obj["e"] =  prefix + entity_type_arr[1].rstrip("]") + "[" + entity_type_arr[0][2:] + "]"
        else:
            new_obj["e"] =  entity_type_arr[1].rstrip("]") + "[" + entity_type_arr[0] + "]"
    return new_obj


def extract_main_entity(results,server_index,pos_index):
    main_entity = results[server_index]["ner"][pos_index]["e"].split('[')[0]
    main_entity = prefix_strip(main_entity)
    return main_entity


def get_span_info(results,server_index,term_index,terms_count):
    pos_index = str(term_index + 1)
    entity = results[server_index]["ner"][pos_index]["e"]
    span_count = 1
    assert(not entity.startswith("I_"))
    if (entity.startswith("B_")):
        term_index += 1
        while(term_index < terms_count):
            pos_index = str(term_index + 1)
            entity = results[server_index]["ner"][pos_index]["e"]
            if (entity == "O"):
                break
            span_count += 1
            term_index += 1
    return span_count

def  is_included_in_server_entities(predictions,s_arr):
    for entity in predictions:
        if ((entity not in s_arr["precedence"]) and (entity not in s_arr["common"])): #treat the presence of an entity in common as a cross over
            return False
    return True



def tag_interactive():
    obj = AggregateNER()
    while True:
        print("Enter text with entity for masked position")
        inp = input()
        if (inp == "q" or inp  == "quit"):
            break
        results = obj.fetch_all(inp)
        print(json.dumps(results,indent=4))

def gen_ner_output(results,fp):
    ner_dict = results["ensembled_ner"]
    for key in ner_dict:
        node = ner_dict[key]
        print(node["term"] + " " + node["e"])
        fp.write(node["term"] + " " + node["e"] + '\n')
    fp.write("\n")
    fp.flush()

def batch_mode(inp_file):
    obj = AggregateNER()
    count = 1
    ner_fp = open(NER_OUTPUT_FILE,"w")
    with open(inp_file) as fp:
        for line in fp:
            line = line.rstrip('\n')
            print(str(count) + "]","Input:",line)
            results = obj.fetch_all(line)
            count += 1
            gen_ner_output(results,ner_fp)
            #print(json.dumps(results,indent=4))
            #pdb.set_trace()


canned_sentences = [
    "Ajit Rajasekharan is an engineer at nFerence",
    "Bio-Techne's genomic tools include advanced tissue-based in-situ hybridization assays (ISH) for research and clinical use, sold under the ACD:__entity__ brand as well as a portfolio of clinical molecular diagnostic oncology assays, including the IntelliScore test (EPI) for prostate cancer diagnosis",
    "I admire my Bari:__entity__ roommates and wish everyone a Happy Casimir:__entity__ Pulaski:__entity__  Day:__entity__",
    "Currently, there are no approved:__entity__ therapies available for CML:__entity__ patients who fail dasatinib:__entity__ or nilotinib:__entity__ in second line",
    "In the LASOR:__entity__ trial:__entity__ , increasing daily imatinib dose from 400 to 600mg induced MMR at 12 and 24 months in 25% and 36% of the patients, respectively, who had suboptimal cytogenetic responses",
    "It is also indicated for the treatment of patients with locally advanced, unresectable or metastatic gastrointestinal stromal tumor (GIST) who have been previously treated with imatinib mesylate and sunitinib:__entity__ malate:__entity__",
    "Ajit Rajasekharan is an engineer at nFerence:__entity__",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy for AML:__entity__",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy in New:__entity__ York:__entity__",
    "Parkinson who lives in Cambridge has been diagnosed with Parkinson's",
    "A non-intrusive sleep apnea detection system using a C-Band:__entity__ channel sensing technique is proposed to monitor sleep apnea syndrome in real time.",
    "Deletion of residues 371–375 from full-length CFTR caused a severe folding defect, resulting in little to no mature form ( C-band:__entity__ ) and eliminating sensitivity to VX-809",
    "New C band markers of human chromosomes: C band position variants.",
    "I thank my Bari:__entity__ friends and wish everyone a Happy Casimir:__entity__ Pulaski:__entity__  Day:__entity__",
    "In 2008, Microbix completed the acquisition of all urokinase:__entity__ assets:__entity__ from ImaRx Therapeutics, Inc., making Microbix the only worldwide source of low-molecular-weight urokinase .",
    "In 2008, Microbix completed the acquisition of all urokinase:__entity__ assets:__entity__ from ImaRx Therapeutics, Inc., making Microbix the only worldwide source of low-molecular-weight urokinase:__entity__ .",
    "Tonsillitis is a type of:__entity__  pharyngitis that:__entity__  typically comes:__entity__  on fast (rapid onset). Symptoms may include sore throat:__entity__  , fever, enlargement of:__entity__  the tonsils, trouble swallowing, and large lymph nodes around the neck. Complications include peritonsillar abscess",
    "Ajit:__entity__ Rajasekharan is an engineer",
    "Ajit:__entity__ Rajasekharan:__entity__ is an engineer",
    "Mesothelioma is caused by exposure to asbestos:__entity__",
    "Parkinson:__entity__ who lives in Cambridge:__entity__ has been diagnosed with Parkinson's:__entity__",
    "Her hypophysitis secondary to ipilimumab:__entity__ was well managed with supplemental hormones",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy in New:__entity__ York:__entity__",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy for AML:__entity__",
    "Her hypophysitis:__entity__ secondary to ipilimumab:__entity__ was well managed with supplemental:__entity__ hormones:__entity__",
    "ajit:__entity__ rajasekharan:__entity__ is an engineer:__entity__ at nFerence:__entity__",
    "Imatinib:__entity__ mesylate:__entity__ is a kinase inhibitor used to treat adults and pediatric patients with Philadelphia + chronic myeloid leukemia (Ph+ CML) and other FDA approved indications.",
    "Imatinib:__entity__ mesylate:__entity__ is a drug and is used to treat nsclc:__entity__",
    "Ajit Rajasekharan is an engineer",
    "New C band markers of human chromosomes  C band position variants",
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
    "After studies at Hofstra University , He worked for New York Telephone before He was elected to the New York State Assembly to represent the 16th District in Northwest Nassau County ",
    "Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",
    "Everyday he rode his bicycle from Rajakilpakkam to Tambaram",
    "If he loses Saturday , it could devalue his position as one of the world 's great boxers , \" Panamanian Boxing Association President Ramon     Manzanares said .",
    "West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",
    "they are his friends ",
    "they flew from Boston to Rio De Janiro and had a mocha",
    "he flew from Boston to Rio De Janiro and had a mocha",
    "X,Y,Z are medicines"
]

def test_canned_sentences():
    obj = AggregateNER()
    for line in canned_sentences:
        results = obj.fetch_all(line)
        print("Input:",line)
        print(json.dumps(results["ensembled_ner"],indent=4))
        pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main NER for a single model ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default=DEFAULT_TEST_BATCH_FILE,help='Input file for batch run option')
    parser.add_argument('-option', action="store", dest="option",default="canned",help='Valid options are canned,batch,interactive. canned - test few canned sentences used in medium artice. batch - tag sentences in input file. Entities to be tagged are determing used POS tagging to find noun phrases.interactive - input one sentence at a time')
    results = parser.parse_args()

    if (results.option == "interactive"):
        tag_interactive()
    elif (results.option == "batch"):
        if (len(results.input) == 0):
            print("Input file needs to be specified")
        else:
            print("Running Batch mode")
            batch_mode(results.input)
    else:
        print("Running canned test mode")
        test_canned_sentences()
