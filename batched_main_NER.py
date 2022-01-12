import pdb
import config_utils as cf
import requests
import sys
import urllib.parse
import numpy as np
from collections import OrderedDict
import argparse
from ensemble.utils.common import *
import json

#WORD_POS = 1
#TAG_POS = 2
#MASK_TAG = "__entity__"
DISPATCH_MASK_TAG = "entity"
DESC_HEAD = "PIVOT_DESCRIPTORS:"
#TYPE2_AMB = "AMB2-"
TYPE2_AMB = ""
DUMMY_DESCS=10
DEFAULT_ENTITY_MAP = "entity_types_consolidated.txt"

#RESET_POS_TAG='RESET'
SPECIFIC_TAG=":__entity__"


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


#noun_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','POS','CD']
#cap_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','PRP']

def read_common_descs(file_name):
    common_descs = {}
    with open(file_name) as fp:
        for line in fp:
            common_descs[line.strip()] = 1
    print("Common descs for filtering read:",len(common_descs))
    return common_descs

def read_entity_map(file_name):
    emap = {}
    with open(file_name) as fp:
        for line in fp:
            line = line.rstrip('\n')
            entities = line.split()
            if (len(entities) == 1):
                assert(entities[0] not in emap)
                emap[entities[0]] = entities[0]
            else:
                assert(len(entities) == 2)
                entity_arr = entities[1].split('/')
                if (entities[0] not in emap):
                    emap[entities[0]] = entities[0]
                for entity in entity_arr:
                    assert(entity not in emap)
                    emap[entity] = entities[0]
    print("Entity map:",len(emap))
    return emap

class UnsupNER:
    def __init__(self):
        print("NER service handler started")
        self.pos_server_url  = cf.read_config()["POS_SERVER_URL"]
        self.desc_server_url  = cf.read_config()["DESC_SERVER_URL"]
        self.entity_server_url  = cf.read_config()["ENTITY_SERVER_URL"]
        self.common_descs = read_common_descs(cf.read_config()["COMMON_DESCS_FILE"])
        self.entity_map = read_entity_map(cf.read_config()["EMAP_FILE"])
        self.rfp = open("log_results.txt","a")
        self.dfp = open("log_debug.txt","a")
        self.algo_ci_tag_fp = open("algorthimic_ci_tags.txt","a")
        print(self.pos_server_url)
        print(self.desc_server_url)
        print(self.entity_server_url)
        np.set_printoptions(suppress=True) #this suppresses exponential representation when np is used to round
        if (cf.read_config()["SUPPRESS_UNTAGGED"] == "1"):
            self.suppress_untagged = True
        else:
            self.suppress_untagged = False #This is disabled in full debug text mode


    #This is bad hack for prototyping - parsing from text output as opposed to json
    def extract_POS(self,text):
        arr = text.split('\n')
        if (len(arr) > 0):
            start_pos = 0
            for i,line in enumerate(arr):
                if (len(line) > 0):
                    start_pos += 1
                    continue
                else:
                    break
            #print(arr[start_pos:])
            terms_arr = []
            for i,line in enumerate(arr[start_pos:]):
                terms = line.split('\t')
                if (len(terms) == 5):
                    #print(terms)
                    terms_arr.append(terms)
            return terms_arr

    def normalize_casing(self,sent):
        sent_arr = sent.split()
        ret_sent_arr = []
        for i,word in enumerate(sent_arr):
            if (len(word) > 1):
                norm_word = word[0] + word[1:].lower()
            else:
                norm_word = word[0]
            ret_sent_arr.append(norm_word)
        return ' '.join(ret_sent_arr)

    #Full sentence tag call also generates json output.
    def tag_sentence_service(self,text,full_sentence_tag):
        if (full_sentence_tag):
            ret_str = self.tag_sentence(text,self.rfp,self.dfp,True)
        else:
            entity_arr,span_arr,terms_arr,ner_str,debug_str = self.tag_sentence(text,self.rfp,self.dfp,False)
            ret_str = ner_str + "\nDEBUG_OUTPUT\n" + '\n'.join(debug_str)
        return ret_str

    def dictify_ner_response(self,ner_str):
        arr = ner_str.split('\n')
        ret_dict = OrderedDict()
        count = 1
        ref_indices_arr = []
        for line in arr:
            terms = line.split()
            if (len(terms) == 2):
                ret_dict[count] = {"term":terms[0],"e":terms[1]}
                if (terms[1] != "O" and terms[1].startswith("B_")):
                        ref_indices_arr.append(count)
                count += 1
            elif (len(terms) == 1):
                ret_dict[count] = {"term":"empty","e":terms[0]}
                if (terms[0] != "O" and terms[0].startswith("B_")):
                        ref_indices_arr.append(count)
                count += 1
                if (len(ret_dict) > 3):  #algorithmic harvesting of CI labels for human verification and adding to bootstrap list
                    self.algo_ci_tag_fp.write("SENT:" + ner_str.replace('\n',' ') + "\n")
                    out = terms[0].replace('[',' ').replace(']','').split()[-1]
                    out = '_'.join(out.split('_')[1:]) if out.startswith("B_") else out
                    print(out)
                    self.algo_ci_tag_fp.write(ret_dict[count-2]["term"] + " " + out + "\n")
                    self.algo_ci_tag_fp.flush()
            else:
                assert(len(terms) == 0) #If not empty something is not right
        return ret_dict,ref_indices_arr

    def blank_entity_sentence(self,sent,dfp):
        value = True if sent.endswith(" :__entity__\n") else False
        if (value == True):
            print("\n\n**************** Skipping CI prediction in pooling for sent:",sent)
            dfp.write("\n\n**************** Skipping CI prediction in pooling for sent:" + sent + "\n")
        return value

    def pool_confidences(self,ci_entities,ci_confidences,ci_subtypes,cs_entities,cs_confidences,cs_subtypes,debug_str_arr,sent,dfp):
        main_classes = {}
        assert(len(cs_entities) ==  len(cs_confidences))
        assert(len(cs_subtypes) ==  len(cs_entities))
        assert(len(ci_entities) ==  len(ci_confidences))
        assert(len(ci_subtypes) ==  len(ci_entities))
        #Pool entity classes across CI and CS
        is_blank_statement =  self.blank_entity_sentence(sent,dfp)  #Do not pool CI confidences of the sentences of the form " is a entity". These sentences are sent for purely algo harvesting of CS terms. CI predictions will add noise.
        if (not is_blank_statement):  #Do not pool CI confidences of the sentences of the form " is a entity". These sentences are sent for purely algo harvesting of CS terms. CI predictions will add noise.
            for e,c in zip(ci_entities,ci_confidences):
                e_base = e.split('[')[0]
                main_classes[e_base] = float(c)
        for e,c in zip(cs_entities,cs_confidences):
            e_base = e.split('[')[0]
            if (e_base in main_classes):
                main_classes[e_base] += float(c)
            else:
                main_classes[e_base] = float(c)
        final_sorted_d = OrderedDict(sorted(main_classes.items(), key=lambda kv: kv[1], reverse=True))
        main_dist = self.convert_positive_nums_to_dist(final_sorted_d)
        main_classes_arr = list(final_sorted_d.keys())
        #print("\nIn pooling confidences")
        #print(main_classes_arr)
        #print(main_dist)
        #Pool subtypes across CI and CS for a particular entity class
        subtype_factors = {}
        for e_class in final_sorted_d:
            if e_class in cs_subtypes:
                stypes = cs_subtypes[e_class]
                if (e_class not in subtype_factors):
                    subtype_factors[e_class] = {}
                for st in stypes:
                    if (st in subtype_factors[e_class]):
                        subtype_factors[e_class][st] += stypes[st]
                    else:
                        subtype_factors[e_class][st] = stypes[st]
            if (is_blank_statement):
                continue
            if e_class in ci_subtypes:
                stypes = ci_subtypes[e_class]
                if (e_class not in subtype_factors):
                    subtype_factors[e_class] = {}
                for st in stypes:
                    if (st in subtype_factors[e_class]):
                        subtype_factors[e_class][st] += stypes[st]
                    else:
                        subtype_factors[e_class][st] = stypes[st]
        sorted_subtype_factors = {}
        for e_class in subtype_factors:
            stypes = subtype_factors[e_class]
            final_sorted_d = OrderedDict(sorted(stypes.items(), key=lambda kv: kv[1], reverse=True))
            stypes_dist = self.convert_positive_nums_to_dist(final_sorted_d)
            stypes_class_arr = list(final_sorted_d.keys())
            sorted_subtype_factors[e_class] = {"stypes":stypes_class_arr,"dist":stypes_dist}
        pooled_results = OrderedDict()
        assert(len(main_classes_arr) == len(main_dist))
        d_str_arr = []
        d_str_arr.append("\n***CONSOLIDATED ENTITY:")
        for e,c in zip(main_classes_arr,main_dist):
            pooled_results[e] = {"e":e,"confidence":c}
            d_str_arr.append(e + " " + str(c))
            stypes_dict = sorted_subtype_factors[e]
            pooled_st = OrderedDict()
            for st,sd in zip(stypes_dict["stypes"],stypes_dict["dist"]):
                pooled_st[st] = sd
            pooled_results[e]["stypes"] = pooled_st
        debug_str_arr.append(' '.join(d_str_arr))
        print(' '.join(d_str_arr))
        return pooled_results









    def init_entity_info(self,entity_info_dict,index):
        curr_term_dict = OrderedDict()
        entity_info_dict[index] = curr_term_dict
        curr_term_dict["ci"] = OrderedDict()
        curr_term_dict["ci"]["entities"] = []
        curr_term_dict["ci"]["descs"] = []
        curr_term_dict["cs"] = OrderedDict()
        curr_term_dict["cs"]["entities"] = []
        curr_term_dict["cs"]["descs"] = []




    #This now does specific tagging if there is a __entity__ in sentence; else does full tagging. TBD.
    #TBD. Make response params same regardlesss of output format. Now it is different
    def tag_sentence(self,sent,rfp,dfp,json_output):
        print("Input: ", sent)
        dfp.write("\n\n++++-------------------------------\n")
        dfp.write("NER_INPUT: " + sent + "\n")
        debug_str_arr = []
        entity_info_dict = OrderedDict()
        url = self.desc_server_url  + sent.replace('"','\'')
        r = self.dispatch_request(url)
        if (r is None):
            print("Empty response. Desc server is probably down: ",self.desc_server_url)
            return json.loads("[]")
        main_obj = json.loads(r.text)
        #print(json.dumps(main_obj,indent=4))
        #Find CI predictions for ALL masked predictios in sentence
        ci_predictions,orig_ci_entities = self.find_ci_entities(main_obj,debug_str_arr,entity_info_dict) #ci_entities is the same info as ci_predictions except packed differently for output
        #Find CS predictions for ALL masked predictios in sentence. Use the CI predictions from previous step to
        #pool
        detected_entities_arr,ner_str,full_pooled_results,orig_cs_entities = self.find_cs_entities(sent,main_obj,rfp,dfp,debug_str_arr,ci_predictions,entity_info_dict)
        assert(len(detected_entities_arr) == len(entity_info_dict))
        print("--------")
        if (json_output):
            if (len(detected_entities_arr) != len(entity_info_dict)):
                if (len(entity_info_dict) == 0):
                    self.init_entity_info(entity_info_dict,index)
                    entity_info_dict[1]["cs"]["entities"].append([{"e":"O","confidence":1}])
                    entity_info_dict[1]["ci"]["entities"].append([{"e":"O","confidence":1}])
            ret_dict,ref_indices_arr  = self.dictify_ner_response(ner_str) #Convert ner string to a dictionary for json output
            assert(len(ref_indices_arr)  == len(detected_entities_arr))
            assert(len(entity_info_dict)  == len(detected_entities_arr))
            cs_aux_dict = OrderedDict()
            ci_aux_dict = OrderedDict()
            cs_aux_orig_entities = OrderedDict()
            ci_aux_orig_entities = OrderedDict()
            pooled_pred_dict = OrderedDict()
            count = 0
            assert(len(full_pooled_results) == len(detected_entities_arr))
            assert(len(full_pooled_results) == len(orig_cs_entities))
            assert(len(full_pooled_results) == len(orig_ci_entities))
            for e,c,p,o,i in zip(detected_entities_arr,entity_info_dict,full_pooled_results,orig_cs_entities,orig_ci_entities):
                val = entity_info_dict[c]
                #cs_aux_dict[ref_indices_arr[count]] = {"e":e,"cs_distribution":val["cs"]["entities"],"cs_descs":val["cs"]["descs"]}
                pooled_pred_dict[ref_indices_arr[count]] = {"e": e, "cs_distribution": list(p.values())}
                cs_aux_dict[ref_indices_arr[count]] = {"e":e,"cs_descs":val["cs"]["descs"]}
                #ci_aux_dict[ref_indices_arr[count]] = {"ci_distribution":val["ci"]["entities"],"ci_descs":val["ci"]["descs"]}
                ci_aux_dict[ref_indices_arr[count]] = {"ci_descs":val["ci"]["descs"]}
                cs_aux_orig_entities[ref_indices_arr[count]] = {"e":e,"cs_distribution": o}
                ci_aux_orig_entities[ref_indices_arr[count]] = {"e":e,"cs_distribution": i}
                count += 1
            #print(ret_dict)
            #print(aux_dict)
            final_ret_dict = {"total_terms_count":len(ret_dict),"detected_entity_phrases_count":len(detected_entities_arr),"ner":ret_dict,"entity_distribution":pooled_pred_dict,"cs_prediction_details":cs_aux_dict,"ci_prediction_details":ci_aux_dict,"orig_cs_prediction_details":cs_aux_orig_entities,"orig_ci_prediction_details":ci_aux_orig_entities,"debug":debug_str_arr}
            json_str = json.dumps(final_ret_dict,indent = 4)
            #print (json_str)
            #with open("single_debug.txt","w") as fp:
            #    fp.write(json_str)

            dfp.write('\n'.join(debug_str_arr))
            dfp.write("\n\nEND-------------------------------\n")
            dfp.flush()
            return json_str
        else:
            print(detected_entities_arr)
            debug_str_arr.append("NER_FINAL_RESULTS: " + ' '.join(detected_entities_arr))
            print("--------")
            dfp.write('\n'.join(debug_str_arr))
            dfp.write("\n\nEND-------------------------------\n")
            dfp.flush()
            return detected_entities_arr,span_arr,terms_arr,ner_str,debug_str_arr

    def masked_word_first_letter_capitalize(self,entity):
        arr = entity.split()
        ret_arr = []
        for term in arr:
            if (len(term) > 1 and term[0].islower() and term[1].islower()):
                ret_arr.append(term[0].upper() + term[1:])
            else:
                ret_arr.append(term)
        return ' '.join(ret_arr)


    def gen_single_phrase_sentences(self,terms_arr,masked_sent_arr,span_arr,rfp,dfp):
        sentence_template = "%s is a entity"
        print(span_arr)
        sentences = []
        singleton_spans_arr  = []
        run_index = 0
        entity  = ""
        singleton_span = []
        while (run_index < len(span_arr)):
            if (span_arr[run_index] == 1):
                while (run_index < len(span_arr)):
                    if (span_arr[run_index] == 1):
                        #print(terms_arr[run_index][WORD_POS],end=' ')
                        if (len(entity) == 0):
                            entity = terms_arr[run_index][WORD_POS]
                        else:
                            entity = entity + " " + terms_arr[run_index][WORD_POS]
                        singleton_span.append(1)
                        run_index += 1
                    else:
                        break
                #print()
                for i in sentence_template.split():
                    if (i != "%s"):
                        singleton_span.append(0)
                entity = self.masked_word_first_letter_capitalize(entity)
                sentence = sentence_template % entity
                sentences.append(sentence)
                singleton_spans_arr.append(singleton_span)
                print(sentence)
                print(singleton_span)
                entity = ""
                singleton_span = []
            else:
                run_index += 1
        return sentences,singleton_spans_arr


    def find_ci_entities(self,main_obj,debug_str_arr,entity_info_dict):
        ci_predictions = []
        orig_ci_confidences = []
        term_index = 1
        batch_obj = main_obj["descs_and_entities"]
        for key in batch_obj:
            masked_sent = batch_obj[key]["ci_prediction"]["sentence"]
            print("\n**CI: ", masked_sent)
            debug_str_arr.append(masked_sent)
            #entity_info_dict["masked_sent"].append(masked_sent)
            inp_arr = batch_obj[key]["ci_prediction"]["descs"]
            descs = self.get_descriptors_for_masked_position(inp_arr)
            self.init_entity_info(entity_info_dict,term_index)
            entities,confidences,subtypes = self.get_entities_for_masked_position(inp_arr,descs,debug_str_arr,entity_info_dict[term_index]["ci"])
            ci_predictions.append({"entities":entities,"confidences":confidences,"subtypes":subtypes})
            orig_ci_confidences.append(self.pack_confidences(entities,confidences))             #this is sent for ensemble server to detect cross predictions. CS predicitons are more reflective of cross over than consolidated predictions, since CI may overwhelm CS
            term_index += 1
        return ci_predictions,orig_ci_confidences


    def pack_confidences(self,cs_entities,cs_confidences):
        assert(len(cs_entities) == len(cs_confidences))
        orig_cs_arr = []
        for e,c in zip(cs_entities,cs_confidences):
            print(e,c)
            e_split = e.split('[')
            e_main = e_split[0]
            if (len(e_split) > 1):
                e_sub = e_split[1].split(',')[0].rstrip(']')
                if (e_main != e_sub):
                    e = e_main + '[' + e_sub + ']'
                else:
                    e = e_main
            else:
                e = e_main
            orig_cs_arr.append({"e":e,"confidence":c})
        return orig_cs_arr


    #We have multiple masked versions of a single sentence. Tag each one of them
    #and create a complete tagged version for a sentence
    def find_cs_entities(self,sent,main_obj,rfp,dfp,debug_str_arr,ci_predictions,entity_info_dict):
        #print(sent)
        batch_obj = main_obj["descs_and_entities"]
        dfp.write(sent + "\n")
        term_index = 1
        detected_entities_arr = []
        full_pooled_results = []
        orig_cs_confidences = []
        for index,key in enumerate(batch_obj):
            position_info = batch_obj[key]["cs_prediction"]["descs"]
            ci_entities = ci_predictions[index]["entities"]
            ci_confidences = ci_predictions[index]["confidences"]
            ci_subtypes = ci_predictions[index]["subtypes"]
            debug_str_arr.append("\n++++++ nth Masked term  : " + key)
            #dfp.write(key + "\n")
            masked_sent = batch_obj[key]["cs_prediction"]["sentence"]
            print("\n**CS: ",masked_sent)
            descs = self.get_descriptors_for_masked_position(position_info)
            #dfp.write(str(descs) + "\n")
            if (len(descs) > 0):
                cs_entities,cs_confidences,cs_subtypes = self.get_entities_for_masked_position(position_info,descs,debug_str_arr,entity_info_dict[term_index]["cs"])
            else:
                cs_entities = []
                cs_confidences = []
                cs_subtypes = []
            #dfp.write(str(cs_entities) + "\n")
            pooled_results = self.pool_confidences(ci_entities,ci_confidences,ci_subtypes,cs_entities,cs_confidences,cs_subtypes,debug_str_arr,sent,dfp)
            self.fill_detected_entities(detected_entities_arr,pooled_results) #just picks the top prediction
            full_pooled_results.append(pooled_results)
            orig_cs_confidences.append(self.pack_confidences(cs_entities,cs_confidences))             #this is sent for ensemble server to detect cross predictions. CS predicitons are more reflective of cross over than consolidated predictions, since CI may overwhelm CS
            #self.old_resolve_entities(i,singleton_entities,detected_entities_arr) #This decides how to pick entities given CI and CS predictions
            term_index += 1
        #out of the full loop over sentences. Now create NER sentence
        terms_arr = main_obj["terms_arr"]
        span_arr = main_obj["span_arr"]
        ner_str = self.emit_sentence_entities(sent,terms_arr,detected_entities_arr,span_arr,rfp) #just outputs results in NER Conll format
        dfp.flush()
        return detected_entities_arr,ner_str,full_pooled_results,orig_cs_confidences


    def fill_detected_entities(self,detected_entities_arr,entities):
        if (len(entities) > 0):
            top_e_class = next(iter(entities))
            top_subtype = next(iter(entities[top_e_class]["stypes"]))
            if (top_e_class != top_subtype):
                top_prediction = top_e_class + "[" + top_subtype + "]"
            else:
                top_prediction = top_e_class
            detected_entities_arr.append(top_prediction)
        else:
            detected_entities_arr.append("OTHER")


    def fill_detected_entities_old(self,detected_entities_arr,entities,pan_arr):
        entities_dict = {}
        count = 1
        for i in entities:
            cand = i.split("-")
            for j in cand:
                terms = j.split("/")
                for k in terms:
                    if (k not in entities_dict):
                        entities_dict[k] = 1.0/count
                    else:
                        entities_dict[k] += 1.0/count
            count += 1
        final_sorted_d = OrderedDict(sorted(entities_dict.items(), key=lambda kv: kv[1], reverse=True))
        first = "OTHER"
        for first in final_sorted_d:
            break
        detected_entities_arr.append(first)

    #Contextual entity is picked as first candidate before context independent candidate
    def old_resolve_entities(self,index,singleton_entities,detected_entities_arr):
        if (singleton_entities[index].split('[')[0] != detected_entities_arr[index].split('[')[0]):
            if (singleton_entities[index].split('[')[0] != "OTHER" and detected_entities_arr[index].split('[')[0] != "OTHER"):
                detected_entities_arr[index] = detected_entities_arr[index] + "/" +  singleton_entities[index]
            elif (detected_entities_arr[index].split('[')[0] == "OTHER"):
                detected_entities_arr[index] =  singleton_entities[index]
            else:
                pass
        else:
           #this is the case when both CI and CS entity type match. Since the subtypes are already ordered, just merge(CS/CI,CS/CI...) the two picking unique subtypes
            main_entity = detected_entities_arr[index].split('[')[0]
            cs_arr = detected_entities_arr[index].split('[')[1].rstrip(']').split(',')
            ci_arr = singleton_entities[index].split('[')[1].rstrip(']').split(',')
            cs_arr_len  = len(cs_arr)
            ci_arr_len  = len(ci_arr)
            max_len = ci_arr_len if ci_arr_len > cs_arr_len else cs_arr_len
            merged_unique_subtype_dict = OrderedDict()
            for i in range(cs_arr_len):
                if (i < cs_arr_len and cs_arr[i] not in merged_unique_subtype_dict):
                    merged_unique_subtype_dict[cs_arr[i]] = 1
                if (i < ci_arr_len and ci_arr[i] not in merged_unique_subtype_dict):
                    merged_unique_subtype_dict[ci_arr[i]] = 1
            new_subtypes_str = ','.join(list(merged_unique_subtype_dict.keys()))
            detected_entities_arr[index] =  main_entity + '[' + new_subtypes_str + ']'






    def emit_sentence_entities(self,sent,terms_arr,detected_entities_arr,span_arr,rfp):
        print("Final result")
        ret_str = ""
        for i,term in enumerate(terms_arr):
            print(term,' ',end='')
        print()
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
            rfp.write(terms_arr[i] + ' ' + tag + "\n")
            ret_str = ret_str + terms_arr[i] + ' ' + tag + "\n"
            print(tag + ' ',end='')
            i += 1
        print()
        rfp.write("\n")
        ret_str += "\n"
        rfp.flush()
        return ret_str





    def get_descriptors_for_masked_position(self,inp_arr):
        desc_arr = []
        for i in range(len(inp_arr)):
            desc_arr.append(inp_arr[i]["desc"])
            desc_arr.append(inp_arr[i]["v"])
        return desc_arr

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

    def convert_positive_nums_to_dist(self,final_sorted_d):
        factors = list(final_sorted_d.values()) #convert dict values to an array
        factors = list(map(float, factors))
        total = float(sum(factors))
        if (total == 0):
            total = 1
            factors[0] = 1 #just make the sum 100%. This a boundary case for numbers for instance
        factors = np.array(factors)
        #factors = softmax(factors)
        factors = factors/total
        factors = np.round(factors,4)
        return factors

    def get_desc_weights_total(self,count,desc_weights):
        i = 0
        total = 0
        while (i < count):
            total += float(desc_weights[i+1])
            i += 2
        total = 1 if total == 0 else total
        return total


    def aggregate_entities(self,entities,desc_weights,debug_str_arr,entity_info_dict_entities):
        ''' Given a masked position, whose entity we are trying to determine,
            First get descriptors for that postion 2*N array [desc1,score1,desc2,score2,...]
            Then for each descriptor, get entity predictions which is an array 2*N of the form [e1,score1,e2,score2,...] where e1 could be DRUG/DISEASE and score1 is 10/8 etc.
            In this function we aggregate each unique entity prediction (e.g. DISEASE) by summing up its weighted scores across all N predictions.
            The result factor array is normalized to create a probability distribution
        '''
        count = len(entities)
        assert(count %2 == 0)
        aggregate_entities = {}
        i = 0
        subtypes = {}
        while (i < count):
            #entities[i] contains entity names and entities[i+] contains counts. Example PROTEIN/GENE/PERSON is i and 10/4/7 is i+1
            curr_counts = entities[i+1].split('/') #this is one of the N predictions - this single prediction is itself  a list of entities
            trunc_e,trunc_counts = self.map_entities(entities[i].split('/'),curr_counts,subtypes) # Aggregate the subtype entities for this predictions. Subtypes aggregation is **across** the N predictions
                                                                                    #Also trunc_e contains the consolidated entity names. 
            assert(len(trunc_e) <= len(curr_counts)) # can be less if untagged is skipped
            assert(len(trunc_e) == len(trunc_counts))
            trunc_counts = softmax(trunc_counts) #this normalization is done to reduce the effect of absolute count of certain labeled entities, while aggregating the entity vectors across descriptors
            curr_counts_sum = sum(map(int,trunc_counts)) #Using truncated count
            curr_counts_sum = 1 if curr_counts_sum == 0 else curr_counts_sum
            for j in range(len(trunc_e)): #this is iterating through the current instance  of all *consolidated* tagged entity predictons  (that is except UNTAGGED_ENTITY)
                if (self.skip_untagged(trunc_e[j])):
                    continue
                if (trunc_e[j] not in aggregate_entities):
                    aggregate_entities[trunc_e[j]] = (float(trunc_counts[j]))*float(desc_weights[i+1])
                    #aggregate_entities[trunc_e[j]] = (float(trunc_counts[j])/curr_counts_sum)*float(desc_weights[i+1])
                    #aggregate_entities[trunc_e[j]] = float(desc_weights[i+1])
                else:
                    aggregate_entities[trunc_e[j]] += (float(trunc_counts[j]))*float(desc_weights[i+1])
                    #aggregate_entities[trunc_e[j]] += (float(trunc_counts[j])/curr_counts_sum)*float(desc_weights[i+1])
                    #aggregate_entities[trunc_e[j]] += float(desc_weights[i+1])
            i += 2
        final_sorted_d = OrderedDict(sorted(aggregate_entities.items(), key=lambda kv: kv[1], reverse=True))
        if (len(final_sorted_d) == 0): #Case where all terms are tagged OTHER
            final_sorted_d = {"OTHER":1}
            subtypes["OTHER"] = {"OTHER":1}
        factors = self.convert_positive_nums_to_dist(final_sorted_d)
        ret_entities = list(final_sorted_d.keys())
        confidences = factors.tolist()
        print(ret_entities)
        sorted_subtypes = self.sort_subtypes(subtypes)
        ret_entities = self.update_entities_with_subtypes(ret_entities,sorted_subtypes)
        print(ret_entities)
        debug_str_arr.append(" ")
        debug_str_arr.append(' '.join(ret_entities))
        print(confidences)
        assert(len(confidences) == len(ret_entities))
        arr = []
        for e,c in zip(ret_entities,confidences):
            arr.append({"e":e,"confidence":c})
        entity_info_dict_entities.append(arr)
        debug_str_arr.append(' '.join([str(x) for x in confidences]))
        debug_str_arr.append("\n\n")
        return ret_entities,confidences,subtypes


    def sort_subtypes(self,subtypes):
        sorted_subtypes =  OrderedDict()
        for ent in subtypes:
            final_sorted_d = OrderedDict(sorted(subtypes[ent].items(), key=lambda kv: kv[1], reverse=True))
            sorted_subtypes[ent]  = list(final_sorted_d.keys())
        return sorted_subtypes

    def update_entities_with_subtypes(self,ret_entities,subtypes):
        new_entities = []

        for ent in ret_entities:
            #if (len(ret_entities) == 1):
            #    new_entities.append(ent) #avoid creating a subtype for a single case
            #    return new_entities
            if (ent in subtypes):
                new_entities.append(ent + '[' + ','.join(subtypes[ent]) + ']')
            else:
                new_entities.append(ent)
        return new_entities

    def skip_untagged(self,term):
        if (self.suppress_untagged == True and (term == "OTHER" or term == "UNTAGGED_ENTITY")):
                return True
        return False


    def map_entities(self,arr,counts_arr,subtypes_dict):
        ret_arr = []
        new_counts_arr = []
        for index,term in enumerate(arr):
            if (self.skip_untagged(term)):
                continue
            ret_arr.append(self.entity_map[term])
            new_counts_arr.append(int(counts_arr[index]))
            if (self.entity_map[term] not in subtypes_dict):
                subtypes_dict[self.entity_map[term]] = {}
            if (term not in subtypes_dict[self.entity_map[term]]):
                #subtypes_dict[self.entity_map[i]][i] = 1
                subtypes_dict[self.entity_map[term]][term] = int(counts_arr[index])
            else:
                #subtypes_dict[self.entity_map[i]][i] += 1
                subtypes_dict[self.entity_map[term]][term] += int(counts_arr[index])
        return ret_arr,new_counts_arr

    def get_entities_from_batch(self,inp_arr):
        entities_arr = []
        for i in range(len(inp_arr)):
            entities_arr.append(inp_arr[i]["e"])
            entities_arr.append(inp_arr[i]["e_count"])
        return entities_arr


    def get_entities_for_masked_position(self,inp_arr,descs,debug_str_arr,entity_info_dict):
        entities = self.get_entities_from_batch(inp_arr)
        debug_combined_arr =[]
        desc_arr =[]
        assert(len(descs) %2 == 0)
        assert(len(entities) %2 == 0)
        index = 0
        for d,e in zip(descs,entities):
            p_e =  '/'.join(e.split('/')[:5])
            debug_combined_arr.append(d + " " + p_e)
            if (index % 2 == 0):
                temp_dict = OrderedDict()
                temp_dict["d"] = d
                temp_dict["e"] = e
            else:
                temp_dict["mlm"] = d
                temp_dict["l_score"] = e
                desc_arr.append(temp_dict)
            index += 1
        debug_str_arr.append("\n" + ', '.join(debug_combined_arr))
        print(debug_combined_arr)
        entity_info_dict["descs"] = desc_arr
        #debug_str_arr.append(' '.join(entities))
        assert(len(entities) == len(descs))
        entities,confidences,subtypes = self.aggregate_entities(entities,descs,debug_str_arr,entity_info_dict["entities"])
        return entities,confidences,subtypes


   #This is again a bad hack for prototyping purposes - extracting fields from a raw text output as opposed to a structured output like json
    def extract_descs(self,text):
        arr = text.split('\n')
        desc_arr = []
        if (len(arr) > 0):
            for i,line in enumerate(arr):
                if (line.startswith(DESC_HEAD)):
                    terms = line.split(':')
                    desc_arr = ' '.join(terms[1:]).strip().split()
                    break
        return desc_arr


    def generate_masked_sentences(self,terms_arr):
        size = len(terms_arr)
        sentence_arr = []
        span_arr = []
        i = 0
        while (i < size):
            term_info = terms_arr[i]
            if (term_info[TAG_POS] in noun_tags):
                skip = self.gen_sentence(sentence_arr,terms_arr,i)
                i +=  skip
                for j in range(skip):
                    span_arr.append(1)
            else:
                i += 1
                span_arr.append(0)
        #print(sentence_arr)
        return sentence_arr,span_arr

    def gen_sentence(self,sentence_arr,terms_arr,index):
        size = len(terms_arr)
        new_sent = []
        for prefix,term in enumerate(terms_arr[:index]):
            new_sent.append(term[WORD_POS])
        i = index
        skip = 0
        while (i < size):
            if (terms_arr[i][TAG_POS] in noun_tags):
                skip += 1
                i += 1
            else:
                break
        new_sent.append(MASK_TAG)
        i = index + skip
        while (i < size):
            new_sent.append(terms_arr[i][WORD_POS])
            i += 1
        assert(skip != 0)
        sentence_arr.append(new_sent)
        return skip








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
                #entity_arr,span_arr,terms_arr,ner_str,debug_str = obj.tag_sentence(line,rfp,dfp,False) # False for json output
                json_str = obj.tag_sentence(line,rfp,dfp,True) # True for json output
                #print("*******************:",terms_arr[span_arr.index(1)][WORD_POS].rstrip(":"),entity_arr[0])
                #sfp.write(terms_arr[span_arr.index(1)][WORD_POS].rstrip(":") + " " + entity_arr[0] + "\n")
                count += 1
                sfp.flush()
                #pdb.set_trace()
    rfp.close()
    sfp.close()
    dfp.close()




test_arr = [
"He felt New:__entity__ York:__entity__ has a chance to win this year's competition",
"Ajit rajasekharan is an engineer at nFerence:__entity__",
"Ajit:__entity__ rajasekharan is an engineer:__entity__ at nFerence:__entity__",
"Mesothelioma:__entity__ is caused by exposure to asbestos:__entity__",
"Fyodor:__entity__ Mikhailovich:__entity__ Dostoevsky:__entity__ was treated for Parkinsons",
"Ajit:__entity__ Rajasekharan:__entity__ is an engineer at nFerence",
"A eGFR:__entity__ below 60 indicates chronic kidney disease",
"A eGFR below 60:__entity__ indicates chronic kidney disease",
"A eGFR:__entity__ below 60:__entity__ indicates chronic:__entity__ kidney:__entity__ disease:__entity__",
"Ajit:__entity__ rajasekharan is an engineer at nFerence",
"Her hypophysitis secondary to ipilimumab was well managed with supplemental hormones",
"In Seattle:__entity__ , Pete Incaviglia 's grand slam with one out in the sixth snapped a tie and lifted the Baltimore Orioles past the Seattle           Mariners , 5-2 .",
"engineer",
"Austin:__entity__ called",
"Paul Erdős died at 83",
"Imatinib mesylate is a drug and is used to treat nsclc",
"In Seattle , Pete Incaviglia 's grand slam with one out in the sixth snapped a tie and lifted the Baltimore Orioles past the Seattle           Mariners , 5-2 .",
"It was Incaviglia 's sixth grand slam and 200th homer of his career .",
"Add Women 's singles , third round Lisa Raymond ( U.S. ) beat Kimberly Po ( U.S. ) 6-3 6-2 .",
"1880s marked the beginning of Jazz",
"He flew from New York to SFO",
"Lionel Ritchie was popular in the 1980s",
"Lionel Ritchie was popular in the late eighties",
"John Doe flew from New York to Rio De Janiro via Miami",
"He felt New York has a chance to win this year's competition",
"Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",
"In humans mutations in Foxp2 leads to verbal dyspraxia",
"The recent spread of Corona virus flu from China to Italy,Iran, South Korea and Japan has caused global concern",
"Hotel California topped the singles chart",
"Elon Musk said Telsa will open a manufacturing plant in Europe",
"He flew from New York to SFO",
"After studies at Hofstra University , He worked for New York Telephone before He was elected to the New York State Assembly to represent the 16th District in Northwest Nassau County ",
"Everyday he rode his bicycle from Rajakilpakkam to Tambaram",
"If he loses Saturday , it could devalue his position as one of the world 's great boxers , \" Panamanian Boxing Association President Ramon     Manzanares said .",
"West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",
"they are his friends ",
"they flew from Boston to Rio De Janiro and had a mocha",
"he flew from Boston to Rio De Janiro and had a mocha",
"X,Y,Z are medicines"]


def test_canned_sentences(obj):
    rfp = open("results.txt","w")
    dfp = open("debug.txt","w")
    for line in test_arr:
        obj = obj.tag_sentence(line,rfp,dfp,True)
        pdb.set_trace()
    rfp.close()
    dfp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main NER for a single model ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input', action="store", dest="input",default="",help='Input file required for run options batch,single')
    parser.add_argument('-option', action="store", dest="option",default="canned",help='Valid options are canned,batch,single. canned - test few canned sentences used in medium artice. batch - tag sentences in input file. Entities to be tagged are determing used POS tagging to find noun phrases. specific - tag specific entities in input file. The tagged word or phrases needs to be of the form w1:__entity_ w2:__entity_ Example:Her hypophysitis:__entity__ secondary to ipilimumab was well managed with supplemental:__entity__ hormones:__entity__')
    results = parser.parse_args()

    obj = UnsupNER()
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
