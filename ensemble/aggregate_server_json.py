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
DEFAULT_THRESHOLD = 1 #1 standard deviation from nean - for cross over prediction

actions_arr = [
        {"url":cf.read_config()["actions_arr"][0]["url"],"desc":cf.read_config()["actions_arr"][0]["desc"], "precedence":cf.read_config()["bio_precedence_arr"],"common":cf.read_config()["common_entities_arr"]},
        {"url":cf.read_config()["actions_arr"][1]["url"],"desc":cf.read_config()["actions_arr"][1]["desc"],"precedence":cf.read_config()["phi_precedence_arr"],"common":cf.read_config()["common_entities_arr"]},
        ]

class AggregateNER:
    def __init__(self):
        self.error_fp = open("failed_queries_log.txt","a")
        self.rfp = open("query_response_log.txt","a")
        self.query_log_fp = open("query_logs.txt","a")
        self.inferred_entities_log_fp = open("inferred_entities_log.txt","a")
        self.threshold = DEFAULT_THRESHOLD #TBD read this from confg. cf.read_config()["CROSS_OVER_THRESHOLD_SIGMA"]
        self.servers  = cf.read_config()["NER_SERVERS"]

    def add_term_punct(self,sent):
        if (len(sent) > 1):
            end_tokens = "!,.:;?"
            last_char = sent[-1]
            if (last_char not in end_tokens): #End all sentences with a period if not already present in sentence.
                sent = sent + ' . '
                print("End punctuated sent:",sent)
        return sent

    def fetch_all(self,inp):
        start = time.time()
        self.query_log_fp.write(inp+"\n")
        self.query_log_fp.flush()
        inp = self.add_term_punct(inp)
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
        return_stat = "Failed" if  len(results["ensembled_ner"]) == 0 else "Success"
        results["stats"] = { "Ensemble server count" : str(len(actions_arr)),
                          "Elapsed time" : str(round(end-start,1)) + " secs", "return_status": return_stat}

        self.rfp.write( "\n" + json.dumps(results,indent=4))
        self.rfp.flush()
        return results


    def get_conflict_resolved_entity(self,results,term_index,terms_count,servers_arr):
        pos_index = str(term_index + 1)
        s1_entity  = extract_main_entity(results,0,pos_index)
        s2_entity  = extract_main_entity(results,1,pos_index)
        span_count1 = get_span_info(results,0,term_index,terms_count)
        span_count2 = get_span_info(results,1,term_index,terms_count)
        if(span_count1 != span_count2):
            print("Both input spans dont match. This is the effect of normalized casing that is model specific. Picking min span length")
            span_count1 = span_count1 if span_count1 <= span_count2 else span_count2
        if (s1_entity == s2_entity):
            server_index = 0 if (s1_entity in servers_arr[0]["precedence"]) else 1
            if (s1_entity != "O"):
                print("Both servers agree on prediction for term:",results[0]["ner"][pos_index]["term"],":",s1_entity)
            return server_index,span_count1,-1
        else:
            print("Servers do not agree on prediction for term:",results[0]["ner"][pos_index]["term"],":",s1_entity,s2_entity)
            #Both the servers dont agree on their predictions. First server is BIO server. Second is PHI
            #Examine both server predictions.
            #Case 1: If just one of them makes a single prediction, then just pick that - it indicates one model is confident while the other isnt.
                #Else.
                # If the top prediction of one of them is a cross prediction, then again drop that prediction and pick the server being cross predicted.
                # Else. Return both predictions, but with the higher confidence prediction first
            #Case 2: Both dont cross predict. Then just return both predictions with higher confidence prediction listed first
            #Cross prediction is checked only for  predictions a server makes ABOVE prediction  mean.
            picked_server_index,cross_prediction_count = self.pick_single_server_if_possible(results,term_index,servers_arr)
        return picked_server_index,span_count1,cross_prediction_count

    def pick_single_server_if_possible(self,results,term_index,servers_arr):
        '''
                Return param : index of picked server
        '''
        pos_index = str(term_index + 1)
        predictions_dict = {}
        orig_cs_predictions_dict = {}
        single_prediction_count = 0
        single_prediction_server_index = -1
        for server_index in range(len(results)):
            if (pos_index in  results[server_index]["entity_distribution"]):
                 predictions = self.get_predictions_above_threshold(results[server_index]["entity_distribution"][pos_index])
                 predictions_dict[server_index]  = predictions  #This is used below to only return top server prediction

                 orig_cs_predictions = self.get_predictions_above_threshold(results[server_index]["orig_cs_prediction_details"][pos_index])
                 orig_cs_predictions_dict[server_index]  = orig_cs_predictions #this is used below for cross prediction determination since it is just a CS prediction
                 single_prediction_count += 1 if (len(orig_cs_predictions) == 1) else 0 
                 if (len(orig_cs_predictions) == 1):
                    single_prediction_server_index = server_index
        if (single_prediction_count == 1):
            is_included = is_included_in_server_entities(orig_cs_predictions_dict[single_prediction_server_index],servers_arr[single_prediction_server_index],False)
            if(is_included == False) :
                print("This is an odd case of single server prediction, that is a cross over")
                ret_index =  0 if single_prediction_server_index == 1 else 1
                return ret_index,-1
            else:
                print("Returning the index of single prediction server")
                return single_prediction_server_index,-1
        elif (single_prediction_count == 2):
            print("Both have single predictions")
            cross_predictions = {}
            cross_prediction_count = 0
            for server_index in range(len(results)):
                if (pos_index in  results[server_index]["entity_distribution"]):
                     is_included = is_included_in_server_entities(orig_cs_predictions_dict[server_index],servers_arr[server_index],False)
                     cross_predictions[server_index] = not is_included
                     cross_prediction_count += 1 if not is_included else 0
            if (cross_prediction_count == 2):
                #this is an odd case of both cross predicting with high confidence. Not sure if we will ever come here.
                print("*********** BOTH servers are cross predicting! ******")
                return self.pick_top_server_prediction(predictions_dict),2
            elif (cross_prediction_count == 0):
                #Neither are cross predecting
                print("*********** BOTH servers have single predictions within their domain - returning both ******")
                return self.pick_top_server_prediction(predictions_dict),2
            else:
                print("Returning just the server that is not cross predicting, dumping the cross prediction")
                ret_index  = 1  if cross_predictions[0] == True else 0 #Given a server cross predicts, return the other server index
                return ret_index,-1
        else:
            print("*** Both servers have multiple predictions above mean")
            #both have multiple predictions above mean
            cross_predictions = {}
            strict_cross_predictions = {}
            cross_prediction_count = 0
            strict_cross_prediction_count = 0
            for server_index in range(len(results)):
                if (pos_index in  results[server_index]["entity_distribution"]):
                     is_included = is_included_in_server_entities(orig_cs_predictions_dict[server_index],servers_arr[server_index],False)
                     strict_is_included = strict_is_included_in_server_entities(orig_cs_predictions_dict[server_index],servers_arr[server_index],False)
                     cross_predictions[server_index] = not is_included
                     strict_cross_predictions[server_index] = not strict_is_included
                     cross_prediction_count += 1 if not is_included else 0
                     strict_cross_prediction_count += 1 if not strict_is_included else 0
            if (cross_prediction_count == 2):
                print("*********** BOTH servers are ALSO cross predicting and have multiple predictions above mean ******")
                return self.pick_top_server_prediction(predictions_dict),2
            elif (cross_prediction_count == 0):
                print("*********** BOTH servers are ALSO predicting within their domain ******")
                #if just one of them is predicting in the common set, then just pick the server that is predicting in its primary set.  
                if (strict_cross_prediction_count == 1):
                    ret_index  = 1  if strict_cross_predictions[0] == True else 0 #Given a server cross predicts, return the other server index
                    return ret_index,-1
                else:
                    return self.pick_top_server_prediction(predictions_dict),2
            else:
                print("Returning just the server that is not cross predicting, dumping the cross prediction. This is mainly to reduce the noise in prefix predictions that show up in CS context predictions")
                ret_index  = 1  if cross_predictions[0] == True else 0 #Given a server cross predicts, return the other server index
                return ret_index,-1
                #print("*********** One of them is also cross predicting  ******")
                #return self.pick_top_server_prediction(predictions_dict),2



    def pick_top_server_prediction(self,predictions_dict):
        '''
        '''        
        if (len(predictions_dict) != 2):
            return 0      
        assert(len(predictions_dict) == 2)
        return 0 if (predictions_dict[0][0]["conf"] >= predictions_dict[1][0]["conf"]) else 1


    def  get_predictions_above_threshold(self,predictions):
        dist = predictions["cs_distribution"]
        sum_predictions = 0
        ret_arr = []
        assert(len(dist) != 0)
        mean_score = 1.0/len(dist) #input is a prob distriubution. so sum is 1
        #sum_deviation = 0
        #for node in dist:
        #    sum_deviation += (mean_score - node["confidence"])*(mean_score - node["confidence"])
        #variance = sum_deviation/len(dist)
        #std_dev = math.sqrt(variance)
        #threshold =  mean_score + std_dev*self.threshold #default is 1 standard deviation from mean
        threshold = mean_score
        pick_count = 1
        for node in dist:
            if (node["confidence"] > threshold):
                ret_arr.append({"e":node["e"],"conf":node["confidence"]})
                pick_count += 1
            else:
                break #this is a reverse sorted list. So no need to check anymore
        return ret_arr

    def check_if_entity_in_arr(self,entity,arr):
        for node in arr:
            if (entity == node["e"]):
                return True
        return False

    def gen_resolved_entity(self,results,server_index,pivot_index,run_index,cross_prediction_count,servers_arr):
        if (cross_prediction_count == 1 or cross_prediction_count == -1):
            #This is the case where we are emitting just one server prediction. In this case, if  CS and consolidated dont match, emit both
            if (pivot_index in results[server_index]["orig_cs_prediction_details"]):
                orig_cs_entity = results[server_index]["orig_cs_prediction_details"][pivot_index]['cs_distribution'][0]
                orig_ci_entity = results[server_index]["orig_ci_prediction_details"][pivot_index]['cs_distribution'][0]
                m1 = orig_cs_entity["e"].split('[')[0]
                m1_ci = orig_ci_entity["e"].split('[')[0]
                is_ci_included = True if (m1_ci in servers_arr[server_index]["precedence"]) else False
                consolidated_entity = results[server_index]["ner"][pivot_index]
                m2,dummy = prefix_strip(consolidated_entity["e"].split('[')[0])
                if (m1 != m2):
                    #if we come here consolidated is not same as cs prediction. So we emit both consolidated and cs
                    ret_obj = results[server_index]["ner"][run_index].copy()
                    dummy,prefix = prefix_strip(ret_obj["e"])
                    n1 = flip_category(orig_cs_entity)
                    n1["e"] = prefix +  n1["e"]
                    n2 = flip_category(consolidated_entity)
                    ret_obj["e"] = n2["e"] + "/" + n1["e"]
                    return ret_obj
                else:
                    #if we come here consolidated is same as cs prediction. So we try to either use ci or the second cs prediction if ci is out of domain
                    if (m1 != m1_ci):
                        #CS and CI are not same
                        if (is_ci_included):
                            #Emity both CS and CI
                            ret_obj = results[server_index]["ner"][run_index].copy()
                            dummy,prefix = prefix_strip(ret_obj["e"])
                            n1 = flip_category(orig_cs_entity)
                            n1["e"] = prefix +  n1["e"]
                            n2 = flip_category(orig_ci_entity)
                            n2["e"] = prefix +  n2["e"]
                            ret_obj["e"] = n1["e"] + "/" + n2["e"]
                            return ret_obj
                        else:
                            #We come here for the case where CI is not in server list. So we pick the second cs as an option if meaningful
                            assert(len(results[server_index]["orig_cs_prediction_details"][pivot_index]['cs_distribution']) >= 2)
                            ret_arr = self.get_predictions_above_threshold(results[server_index]["orig_cs_prediction_details"][pivot_index])
                            orig_cs_second_entity = results[server_index]["orig_cs_prediction_details"][pivot_index]['cs_distribution'][1]
                            m2_cs = orig_cs_second_entity["e"].split('[')[0]
                            is_cs_included = True if (m2_cs in servers_arr[server_index]["precedence"]) else False
                            is_cs_included = True #Disabling cs included check. If prediction above threshold is cross prediction, then letting it through
                            assert (m2_cs != m1)
                            if (is_cs_included and self.check_if_entity_in_arr(m2_cs,ret_arr)):
                                ret_obj = results[server_index]["ner"][run_index].copy()
                                dummy,prefix = prefix_strip(ret_obj["e"])
                                n1 = flip_category(orig_cs_second_entity)
                                n1["e"] = prefix +  n1["e"]
                                n2 = flip_category(orig_cs_entity)
                                n2["e"] = prefix +  n2["e"]
                                ret_obj["e"] = n2["e"] + "/" + n1["e"]
                                return ret_obj
                            else:
                                return flip_category(results[server_index]["ner"][run_index])
                    else:
                        #here cs and ci are same. So use two cs predictions if meaningful
                        if (len(results[server_index]["orig_cs_prediction_details"][pivot_index]['cs_distribution']) >= 2):
                            ret_arr = self.get_predictions_above_threshold(results[server_index]["orig_cs_prediction_details"][pivot_index])
                            orig_cs_second_entity = results[server_index]["orig_cs_prediction_details"][pivot_index]['cs_distribution'][1]
                            m2_cs = orig_cs_second_entity["e"].split('[')[0]
                            is_cs_included = True if (m2_cs in servers_arr[server_index]["precedence"]) else False
                            is_cs_included = True #Disabling cs included check. If prediction above threshold is cross prediction, then letting it through
                            assert (m2_cs != m1)
                            if (is_cs_included and self.check_if_entity_in_arr(m2_cs,ret_arr)):
                                ret_obj = results[server_index]["ner"][run_index].copy()
                                dummy,prefix = prefix_strip(ret_obj["e"])
                                n1 = flip_category(orig_cs_second_entity)
                                n1["e"] = prefix +  n1["e"]
                                n2 = flip_category(orig_cs_entity)
                                n2["e"] = prefix +  n2["e"]
                                ret_obj["e"] = n2["e"] + "/" + n1["e"]
                                return ret_obj
                            else:
                                return flip_category(results[server_index]["ner"][run_index])
                        else:
                                return flip_category(results[server_index]["ner"][run_index])
            else:
                return flip_category(results[server_index]["ner"][run_index])
        else:
            #Case where both servers dont match
            ret_obj = results[server_index]["ner"][run_index].copy()
            #ret_obj["e"] = results[0]["ner"][run_index]["e"] + "/" + results[1]["ner"][run_index]["e"]
            index2 = 1 if  server_index == 0 else 0 #this is the index of the dominant server with hihgher prediction confidence
            n1 = flip_category(results[server_index]["ner"][run_index])
            n2 = flip_category(results[index2]["ner"][run_index])
            ret_obj["e"] = n1["e"] + "/" + n2["e"]
            return ret_obj


    def confirm_same_size_responses(self,sent,results):
     count = 0
     for i in range(len(results)):
         if ("ner" in results[i]):
             ner = results[i]["ner"]
         else:
             print("Server",i," returned invalid response;",results[i])
             self.error_fp.write("Server " + str(i) + " failed for query: " + sent + "\n")
             self.error_fp.flush()
             return 0
         if(count == 0):
             assert(len(ner) > 0)
             count = len(ner)
         else:
             if (count != len(ner)):
                  print("Warning. The return sizes of both servers do not match. This must be truncated sentence, where tokenization causes different length truncations. Using min length") 
                  count  = count if count < len(ner) else len(ner)
     return count


    def get_ensembled_entities(self,sent,results,servers_arr):
        ensembled_ner = OrderedDict()
        orig_cs_predictions = OrderedDict()
        orig_ci_predictions = OrderedDict()
        ensembled_conf =  OrderedDict()
        ambig_ensembled_conf =  OrderedDict()
        ensembled_ci = OrderedDict()
        ensembled_cs = OrderedDict()
        ambig_ensembled_ci = OrderedDict()
        ambig_ensembled_cs = OrderedDict()
        print("Ensemble candidates")
        terms_count =  self.confirm_same_size_responses(sent,results)
        if (terms_count == 0):
            return ensembled_ner,ensembled_conf,ensembled_ci,ensembled_cs,ambig_ensembled_conf,ambig_ensembled_ci,ambig_ensembled_cs,orig_cs_predictions,orig_ci_predictions
        assert(len(servers_arr) == len(results))
        term_index = 0
        while (term_index  < terms_count):
            pos_index = str(term_index + 1)
            assert(len(servers_arr) == 2) #TBD. Currently assumes two servers in prototype to see if this approach works. To be extended to multiple servers
            server_index,span_count,cross_prediction_count = self.get_conflict_resolved_entity(results,term_index,terms_count,servers_arr)
            pivot_index = str(term_index + 1)
            for span_index in range(span_count):
                run_index = str(term_index + 1 + span_index)
                ensembled_ner[run_index] = self.gen_resolved_entity(results,server_index,pivot_index,run_index,cross_prediction_count,servers_arr)
                if (run_index in  results[server_index]["entity_distribution"]):
                    ensembled_conf[run_index] = results[server_index]["entity_distribution"][run_index]
                    ensembled_conf[run_index]["e"] = strip_prefixes(ensembled_ner[run_index]["e"]) #this is to make sure the same tag can be taken from NER result or this structure.
                                                                                   #When both server responses are required, just return the details of first server for now
                    ensembled_ci[run_index] = results[server_index]["ci_prediction_details"][run_index]
                    ensembled_cs[run_index] = results[server_index]["cs_prediction_details"][run_index]
                    orig_cs_predictions[run_index] = results[server_index]["orig_cs_prediction_details"][run_index]
                    orig_ci_predictions[run_index] = results[server_index]["orig_ci_prediction_details"][run_index]

                    if (cross_prediction_count == 0 or cross_prediction_count == 2): #This is an ambiguous prediction. Send both server responses
                        second_server = 1 if server_index == 0 else 1
                        ambig_ensembled_conf[run_index] = results[second_server]["entity_distribution"][run_index]
                        ambig_ensembled_conf[run_index]["e"] = ensembled_ner[run_index]["e"] #this is to make sure the same tag can be taken from NER result or this structure.
                        ambig_ensembled_ci[run_index] = results[second_server]["ci_prediction_details"][run_index]
                if (ensembled_ner[run_index]["e"] != "O"):
                    self.inferred_entities_log_fp.write(results[0]["ner"][run_index]["term"] + " " + ensembled_ner[run_index]["e"]  + "\n")
            term_index += span_count
        self.inferred_entities_log_fp.flush()
        return ensembled_ner,ensembled_conf,ensembled_ci,ensembled_cs,ambig_ensembled_conf,ambig_ensembled_ci,ambig_ensembled_cs,orig_cs_predictions,orig_ci_predictions



    def ensemble_processing(self,sent,results):
        ensembled_ner,ensembled_conf,ci_details,cs_details,ambig_ensembled_conf,ambig_ci_details,ambig_cs_details,orig_cs_predictions,orig_ci_predictions = self.get_ensembled_entities(sent,results,actions_arr)
        final_ner = OrderedDict()
        final_ner["ensembled_ner"] = ensembled_ner
        final_ner["ensembled_prediction_details"] = ensembled_conf
        final_ner["ci_prediction_details"] = ci_details
        final_ner["cs_prediction_details"] = cs_details
        final_ner["ambig_prediction_details_conf"] = ambig_ensembled_conf
        final_ner["ambig_prediction_details_ci"] = ambig_ci_details
        final_ner["ambig_prediction_details_cs"] = ambig_cs_details
        final_ner["orig_cs_prediction_details"] = orig_cs_predictions
        final_ner["orig_ci_prediction_details"] = orig_ci_predictions
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
      try:
          self.results = json.loads(out.text,object_pairs_hook=OrderedDict)
      except:
            print("Empty response from server for input:",self.param)
            self.results =  json.loads("{}",object_pairs_hook=OrderedDict)
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



def prefix_strip(term):
    prefix = ""
    if (term.startswith("B_") or term.startswith("I_")):
        prefix = term[:2]
        term = term[2:]
    return term,prefix

def strip_prefixes(term):
    split_entities = term.split('/')
    if (len(split_entities) == 2):
        term1,dummy = prefix_strip(split_entities[0])
        term2,dummy = prefix_strip(split_entities[1])
        return term1 + '/' + term2
    else:
        assert(len(split_entities)  == 1)
        term1,dummy = prefix_strip(split_entities[0])
        return term1


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
    main_entity,dummy = prefix_strip(main_entity)
    return main_entity


def get_span_info(results,server_index,term_index,terms_count):
    pos_index = str(term_index + 1)
    entity = results[server_index]["ner"][pos_index]["e"]
    span_count = 1
    if (entity.startswith("I_")):
        print("Skipping an I tag for server:",server_index,". This has to be done because of mismatched span because of model specific casing normalization that changes POS tagging. This happens only for sentencees user does not explicirly tag with ':__entity__'")
        return span_count
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

def  is_included_in_server_entities(predictions,s_arr,check_first_only):
    for entity in predictions:
        entity = entity['e'].split('[')[0]
        if ((entity not in s_arr["precedence"]) and (entity not in s_arr["common"])): #do not treat the presence of an entity in common as a cross over
            return False
        if (check_first_only):
            return True #Just check the top prediction for inclusion in the new semantics
    return True

def  strict_is_included_in_server_entities(predictions,s_arr,check_first_only):
    for entity in predictions:
        entity = entity['e'].split('[')[0]
        if ((entity not in s_arr["precedence"])): #do not treat the presence of an entity in common as a cross over
            return False
        if (check_first_only):
            return True #Just check the top prediction for inclusion in the new semantics
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
    print()
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


canned_sentences = [
    "imatinib was used to treat Michael:__entity__ Jackson:__entity__",
    "Mesothelioma is caused by exposure to asbestos:__entity__",
    "It was Incaviglia 's sixth grand slam and 200th homer of his career .",
    "Add Women 's singles , third round Lisa Raymond ( U.S. , beat Kimberly Po ( U.S. , 6-3 6-2 .",
    "1880s marked the beginning of Jazz",
    "He flew from New York to SFO",
    "Lionel Ritchie was popular in the 1980s",
    "Lionel Ritchie was popular in the late eighties",
    "John Doe flew from New York to Rio De Janiro via Miami",
    "Fyodor Mikhailovich Dostoevsky was treated for Parkinsons",
    "In humans mutations in Foxp2 leads to verbal dyspraxia",
    "The recent spread of Corona virus flu from China to Italy,Iran, South Korea and Japan has caused global concern",
    "Hotel California topped the singles chart",
    "After studies at Hofstra University , He worked for New York Telephone before He was elected to the New York State Assembly to represent the 16th District in Northwest Nassau County ",
    "Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",
    "Everyday he rode his bicycle from Rajakilpakkam to Tambaram",
    "If he loses Saturday , it could devalue his position as one of the world 's great boxers , \" Panamanian Boxing Association President Ramon     Manzanares said .",
    "West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",
    "they are his friends ",
    "they flew from Boston to Rio De Janiro and had a mocha",
    "he flew from Boston to Rio De Janiro and had a mocha",
    "X,Y,Z are medicines",
    "The portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy: for AML:__entity__",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical giant Pfizer  said Wednesday",
    "He flew from New York to SFO",
    "He felt New York has a chance:__entity__ to win this year's competition",
    "Lucid:__entity__ shutdown its plant in US and laid off over a thousand employees",
    "Elon Musk said Tesla will open a manufacturing plant in Europe",
    "Elon Musk said Telsa will open a manufacturing plant in Europe",
    "He felt New:__entity__ York:__entity__ has a chance to win this year's competition",
    "He felt New York has a chance to win this year's competition",
    "Parkinson:__entity__ who lives in Cambridge:__entity__ has been diagnosed with Parkinson's:__entity__",
    "The new omicron:__entity__ variant:__entity__ could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical giant Pfizer  said Wednesday",
    "The new:__entity__ omicron:__entity__ variant:__entity__ could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical giant Pfizer  said Wednesday",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives:__entity__ at pharmaceutical giant_ Pfizer said Wednesday",
    "ajit rajasekharan is an engineer at nFerence",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical giant_ Pfizer:__entity__  said Wednesday",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical:__entity__ giant:__entity__ Pfizer:__entity__  said Wednesday",
    "It is also indicated for the treatment of patients with locally advanced, unresectable or metastatic gastrointestinal stromal tumor (GIST) who have been previously treated with imatinib:__entity__ mesylate:__entity__ and sunitinib:__entity__ malate:__entity__",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical giant Pfizer  said Wednesday",
    "The new omicron variant could increase the likelihood that people will need a fourth coronavirus  vaccine dose earlier than expected, executives at pharmaceutical:__entity__ giant:__entity__ Pfizer:__entity__  said Wednesday",
    "The new omicron:__entity__  variant could increase the likelihood that people will need a fourth coronavirus:__entity__  vaccine dose earlier than expected, executives at pharmaceutical giant Pfizer:__entity__  said Wednesday",
    "conditions:__entity__ that can cause pulmonary hypertension include sickle cell disease; pulmonary embolus, which is a type of venous thromboembolism; and chronic obstructive pulmonary disease",
    "conditions that can cause pulmonary hypertension include sickle cell disease; pulmonary embolus, which is a type:__entity__ of venous thromboembolism; and chronic obstructive pulmonary disease",
    "conditions that can cause pulmonary hypertension include sickle cell disease; pulmonary embolus, which is a type of venous thromboembolism; and chronic obstructive pulmonary disease",
    "He fell down and broke his leg:__entity__",
    "He fell down and broke his:__entity__ leg",
    "He fell down and broke:__entity__ his leg",
    "He fell down and:__entity__ broke his leg",
    "He fell:__entity__ down and broke his leg",
    "He fell down:__entity__ and broke his leg",
    "I met my best friend at the Parkinson's",
    "I met my best friend at Parkinson's",
    "Parkinson suffers from Parkinson's",
    "He died at 67 after a prolonged illness",
    "He walked at a constant speed of 3 mph for 45 minutes and then increased his speed by 10% at the very end before relaxing for an hour at the bar for a drink",
    ";",
    "100 ml",
    "1",
    "e",

    "eg",
    "ajit rajasekharan is an engineer at nFerence:__entity__",
    "Imatinib meslyate is used to trear nsclc:__entity__",
    "Imatinib:__entity__ meslyate:__entity__ is used to trear nsclc",
    "imatinib was used to treat Michael:__entity__ Jackson:__entity__",
    "This strong enrichment in ruthenium reaches an apogee at the center of the massive:__entity__ sulphide:__entity__ zone:__entity__ ",
    "Ajit is an engineer:__entity__",
    "the most common cause of pulmonary hypertension is left heart disease. Other:__entity__ conditions:__entity__ that can cause pulmonary hypertension include sickle cell disease; pulmonary embolus, which is a type of venous thromboembolism; and chronic obstructive pulmonary disease",
    "The coronavirus disease  ( COVID-19:__entity__ ) is caused by a virus NOT by bacteria",
    "The coronavirus disease  (COVID-19:__entity__ ) is caused by a virus NOT by bacteria",
    "ajit rajasekharan is an engineer at nFerence",
    "The portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy for AML:__entity__",
    "Omicron:__entity__ live updates: Variant detected in Houston's wastewater",
    "Omicron live updates: Variant detected in Houston's wastewater",
    "Bio-Techne's genomic tools include advanced tissue-based in-situ hybridization assays (ISH) for research and clinical use, sold under the ACD:__entity__ brand as well as a portfolio of clinical molecular diagnostic oncology assays, including the IntelliScore test (EPI) for prostate cancer diagnosis",
    "I admire my Bari:__entity__ roommates and wish everyone a Happy Casimir:__entity__ Pulaski:__entity__  Day:__entity__",
    "Currently, there are no approved:__entity__ therapies available for CML:__entity__ patients who fail dasatinib:__entity__ or nilotinib:__entity__ in second line",
    "In the LASOR:__entity__ trial:__entity__ , increasing daily imatinib dose from 400 to 600mg induced MMR at 12 and 24 months in 25% and 36% of the patients, respectively, who had suboptimal cytogenetic responses",
    "It is also indicated for the treatment of patients with locally advanced, unresectable or metastatic gastrointestinal stromal tumor (GIST) who have been previously treated with imatinib mesylate and sunitinib:__entity__ malate:__entity__",
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
    "Mesothelioma is caused by exposure to asbestos:__entity__",
    "Her hypophysitis secondary to ipilimumab:__entity__ was well managed with supplemental hormones",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy in New:__entity__ York:__entity__",
    "the portfolio manager of the new cryptocurrency firm underwent a bone marrow biopsy for AML:__entity__",
    "Her hypophysitis:__entity__ secondary to ipilimumab:__entity__ was well managed with supplemental:__entity__ hormones:__entity__",
    "Imatinib:__entity__ mesylate:__entity__ is a kinase inhibitor used to treat adults and pediatric patients with Philadelphia + chronic myeloid leukemia (Ph+ CML) and other FDA approved indications.",
    "Imatinib:__entity__ mesylate:__entity__ is a drug and is used to treat nsclc:__entity__",
    "New C band markers of human chromosomes  C band position variants",
    "engineer",
    "Austin called",
    "Her hypophysitis secondary to ipilimumab was well managed with supplemental hormones",
    "Paul Erdős died at 83",
    "In Seattle , Pete Incaviglia 's grand slam with one out in the sixth snapped a tie and lifted the Baltimore Orioles past the Seattle           Mariners , 5-2 .",
    "Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",
    "Input: Coronavirus:__entity__ disease 2019 (COVID-19) is a contagious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first known case was identified in Wuhan, China, in December 2019.[7] The disease has since spread worldwide, leading to an ongoing pandemic.[8]Symptoms of COVID-19 are variable, but often include fever,[9] cough, headache,[10] fatigue, breathing difficulties, and loss of smell and taste.[11][12][13] Symptoms may begin one to fourteen days after exposure to the virus. At least a third of people who are infected do not develop noticeable symptoms.[14] Of those people who develop symptoms noticeable enough to be classed as patients, most (81%) develop mild to moderate symptoms (up to mild pneumonia), while 14% develop severe symptoms (dyspnea, hypoxia, or more than 50% lung involvement on imaging), and 5% suffer critical symptoms (respiratory failure, shock, or multiorgan dysfunction).[15] Older people are at a higher risk of developing severe symptoms. Some people continue to experience a range of effects (long COVID) for months after recovery, and damage to organs has been observed.[16] Multi-year studies are underway to further investigate the long-term effects of the disease.[16]COVID-19 transmits when people breathe in air contaminated by droplets and small airborne particles containing the virus. The risk of breathing these in is highest when people are in close proximity, but they can be inhaled over longer distances, particularly indoors. Transmission can also occur if splashed or sprayed with contaminated fluids in the eyes, nose or mouth, and, rarely, via contaminated surfaces. People remain contagious for up to 20 days, and can spread the virus even if they do not develop symptoms.[17][18]Several testing methods have been developed to diagnose the disease. The standard diagnostic method is by detection of the virus' nucleic acid by real-time reverse transcription polymerase chain reaction (rRT-PCR), transcription-mediated amplification (TMA), or by reverse transcription loop-mediated isothermal amplification (RT-LAMP) from a nasopharyngeal swab.Several COVID-19 vaccines have been approved and distributed in various countries, which have initiated mass vaccination campaigns. Other preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. The use of face masks or coverings has been recommended in public settings to minimize the risk of transmissions. While work is underway to develop drugs that inhibit the virus, the primary treatment is symptomatic. Management involves the treatment of symptoms, supportive care, isolation, and experimental measures."
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
