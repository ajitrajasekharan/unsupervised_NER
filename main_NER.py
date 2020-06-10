import pdb
import config_utils as cf
import requests
import sys
import urllib.parse
from collections import OrderedDict

WORD_POS = 1
TAG_POS = 2
MASK_TAG = "__entity__"
DISPATCH_MASK_TAG = "entity"
DESC_HEAD = "PIVOT_DESCRIPTORS:"
TYPE2_AMB = "AMB2-"
TOPK_DESCS=5

RESET_POS_TAG='RESET'


noun_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','POS','CD']
cap_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','PRP']

def read_common_descs(file_name):
    common_descs = {}
    with open(file_name) as fp:
        for line in fp:
            common_descs[line.strip()] = 1
    print("Common descs for filtering read:",len(common_descs))
    return common_descs

class UnsupNER:
    def __init__(self):
        self.pos_server_url  = cf.read_config()["POS_SERVER_URL"]
        self.desc_server_url  = cf.read_config()["DESC_SERVER_URL"]
        self.entity_server_url  = cf.read_config()["ENTITY_SERVER_URL"]
        self.common_descs = read_common_descs(cf.read_config()["COMMON_DESCS_FILE"])
        print(self.pos_server_url)
        print(self.desc_server_url)
        print(self.entity_server_url)

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

    def capitalize(self,terms_arr):
        for i,term_tag in enumerate(terms_arr):
            #print(term_tag)
            if (term_tag[TAG_POS] in cap_tags):
                word = term_tag[WORD_POS][0].upper() + term_tag[WORD_POS][1:]
                term_tag[WORD_POS] = word
        #print(terms_arr)

    def set_POS_based_on_entities(self,sent):
        terms_arr = []
        sent_arr = sent.split()
        for i,word in enumerate(sent_arr):
            #print(term_tag)
            term_tag = ['-']*5
            if (word.endswith(MASK_TAG)):
                term_tag[TAG_POS] = noun_tags[0]
                term_tag[WORD_POS] = word.replace(MASK_TAG,"").rstrip(":")
            else:
                term_tag[TAG_POS] = RESET_POS_TAG
                term_tag[WORD_POS] = word
            terms_arr.append(term_tag)
        return terms_arr
        #print(terms_arr)

    def filter_common_noun_spans(self,span_arr,masked_sent_arr,terms_arr):
        ret_span_arr = span_arr.copy()
        ret_masked_sent_arr = []
        sent_index = 0
        loop_span_index = 0
        while (loop_span_index < len(span_arr)):
            span_val = span_arr[loop_span_index]
            orig_index = loop_span_index
            if (span_val == 1):
                curr_index = orig_index
                is_all_common = True
                while (curr_index < len(span_arr) and span_arr[curr_index] == 1):
                    term = terms_arr[curr_index]
                    if (term[WORD_POS].lower() not in self.common_descs):
                        is_all_common = False
                    curr_index += 1
                loop_span_index = curr_index #note the loop scan index is updated
                if (is_all_common):
                    curr_index = orig_index
                    print("Filtering common span: ",end='')
                    while (curr_index < len(span_arr) and span_arr[curr_index] == 1):
                        print(terms_arr[curr_index][WORD_POS],' ',end='')
                        ret_span_arr[curr_index] = 0
                        curr_index += 1
                    print()
                    sent_index += 1 # we are skipping a span
                else:
                    ret_masked_sent_arr.append(masked_sent_arr[sent_index])
                    sent_index += 1
            else:
                loop_span_index += 1
        return ret_masked_sent_arr,ret_span_arr

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


    def tag_sentence(self,sent,rfp,dfp):
        sent = self.normalize_casing(sent)
        print("Caps normalized:", sent)
        url = self.pos_server_url  + sent.replace('"','\'')
        r = self.dispatch_request(url)
        terms_arr = self.extract_POS(r.text)
        self.capitalize(terms_arr)
        masked_sent_arr,span_arr = self.generate_masked_sentences(terms_arr)
        masked_sent_arr,span_arr = self.filter_common_noun_spans(span_arr,masked_sent_arr,terms_arr)
        singleton_sentences,singleton_spans_arr = self.gen_single_phrase_sentences(terms_arr,masked_sent_arr,span_arr,rfp,dfp)
        singleton_entities = self.find_singleton_entities(singleton_sentences,singleton_spans_arr)
        self.find_entities(sent,terms_arr,masked_sent_arr,span_arr,singleton_entities,rfp,dfp)
        print("--------")

    def tag_se_in_sentence(self,sent,rfp,dfp):
        sent = self.normalize_casing(sent)
        print("Caps normalized:", sent)
        terms_arr = self.set_POS_based_on_entities(sent)
        masked_sent_arr,span_arr = self.generate_masked_sentences(terms_arr)
        masked_sent_arr,span_arr = self.filter_common_noun_spans(span_arr,masked_sent_arr,terms_arr)
        singleton_sentences,singleton_spans_arr = self.gen_single_phrase_sentences(terms_arr,masked_sent_arr,span_arr,rfp,dfp)
        singleton_entities = self.find_singleton_entities(singleton_sentences,singleton_spans_arr)
        detected_entities_arr = self.find_entities(sent,terms_arr,masked_sent_arr,span_arr,singleton_entities,rfp,dfp)
        print(detected_entities_arr)
        print("--------")
        return detected_entities_arr,span_arr,terms_arr



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


    def find_singleton_entities(self,masked_sent_arr,span_arr):
        detected_entities_arr = []
        for dummy,masked_sent in enumerate(masked_sent_arr):
            print(masked_sent)
            descs = self.get_descriptors_for_masked_position(masked_sent)
            entities = self.get_entities_for_masked_position(descs)
            self.fill_detected_entities(detected_entities_arr,entities,span_arr)
        return detected_entities_arr



    #We have multiple masked versions of a single sentence. Tag each one of them
    #and create a complete tagged version for a sentence
    def find_entities(self,sent,terms_arr,masked_sent_arr,span_arr,singleton_entities,rfp,dfp):
        #print(sent)
        print(span_arr)
        dfp.write(sent + "\n")
        dfp.write(str(span_arr) + "\n")
        detected_entities_arr = []
        for i,masked_sent in enumerate(masked_sent_arr):
            masked_sent = ' '.join(masked_sent)
            print(masked_sent)
            dfp.write(masked_sent + "\n")
            descs = self.get_descriptors_for_masked_position(masked_sent)
            dfp.write(str(descs) + "\n")
            entities = self.get_entities_for_masked_position(descs)
            dfp.write(str(entities) + "\n")
            self.fill_detected_entities(detected_entities_arr,entities,span_arr)
            self.resolve_entities(i,singleton_entities,detected_entities_arr)
        self.emit_sentence_entities(sent,terms_arr,detected_entities_arr,span_arr,rfp)
        dfp.flush()
        return detected_entities_arr

    def fill_detected_entities(self,detected_entities_arr,entities,pan_arr):
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
    def resolve_entities(self,index,singleton_entities,detected_entities_arr):
        if (singleton_entities[index] != detected_entities_arr[index]):
            #detected_entities_arr[index] = TYPE2_AMB +  singleton_entities[index] + "/" + detected_entities_arr[index]
            detected_entities_arr[index] = TYPE2_AMB +   detected_entities_arr[index] + "/" +  singleton_entities[index]


    def emit_sentence_entities(self,sent,terms_arr,detected_entities_arr,span_arr,rfp):
        print("Final result")
        for i,term in enumerate(terms_arr):
            print(term[WORD_POS],' ',end='')
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
            rfp.write(terms_arr[i][WORD_POS] + ' ' + tag + "\n")
            print(tag + ' ',end='')
            i += 1
        print()
        rfp.write("\n")
        rfp.flush()






    def get_descriptors_for_masked_position(self,masked_sent):
        masked_sent = masked_sent.replace(MASK_TAG,DISPATCH_MASK_TAG)
        r = self.dispatch_request(self.desc_server_url+str(masked_sent))
        desc_arr = self.extract_descs(r.text)
        print(desc_arr)
        return desc_arr[:TOPK_DESCS]

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

    def get_entities_for_masked_position(self,descs):
        param = ' '.join(descs)
        r = self.dispatch_request(self.entity_server_url+str(param))
        entities = r.text.split()
        print(entities)
        return entities


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
                entity_arr,span_arr,terms_arr = obj.tag_se_in_sentence(line,rfp,dfp)
                print("*******************:",terms_arr[span_arr.index(1)][WORD_POS].rstrip(":"),entity_arr[0])
                sfp.write(terms_arr[span_arr.index(1)][WORD_POS].rstrip(":") + " " + entity_arr[0] + "\n")
                count += 1
                sfp.flush()
    rfp.close()
    sfp.close()
    dfp.close()


def test_canned_sentences(obj):
    rfp = open("results.txt","w")
    dfp = open("debug.txt","w")
    obj.tag_sentence("engineer",rfp,dfp)
    obj.tag_sentence("Paul Erdős died at 83",rfp,dfp)
    obj.tag_sentence("ajit rajasekharan is an engineer",rfp,dfp)
    obj.tag_sentence("Imatinib mesylate is a drug and is used to treat nsclc",rfp,dfp)
    obj.tag_sentence("In Seattle , Pete Incaviglia 's grand slam with one out in the sixth snapped a tie and lifted the Baltimore Orioles past the Seattle           Mariners , 5-2 .",rfp,dfp)
    obj.tag_sentence("It was Incaviglia 's sixth grand slam and 200th homer of his career .",rfp,dfp)
    obj.tag_sentence("Add Women 's singles , third round Lisa Raymond ( U.S. ) beat Kimberly Po ( U.S. ) 6-3 6-2 .",rfp,dfp)
    obj.tag_sentence("1880s marked the beginning of Jazz",rfp,dfp)
    obj.tag_sentence("He flew from New York to SFO",rfp,dfp)
    obj.tag_sentence("Lionel Ritchie was popular in the 1980s",rfp,dfp)
    obj.tag_sentence("Lionel Ritchie was popular in the late eighties",rfp,dfp)
    obj.tag_sentence("John Doe flew from New York to Rio De Janiro via Miami",rfp,dfp)
    obj.tag_sentence("He felt New York has a chance to win this year's competition",rfp,dfp)
    obj.tag_sentence("He flew from New York to SFO",rfp,dfp)
    obj.tag_sentence("Bandolier - Budgie ' , a free itunes app for ipad , iphone and ipod touch , released in December 2011 , tells the story of the making of Bandolier in the band 's own words - including an extensive audio interview with Burke Shelley",rfp,dfp)
    obj.tag_sentence("Fyodor Mikhailovich Dostoevsky was treated for Parkinsons",rfp,dfp)
    obj.tag_sentence("In humans mutations in Foxp2 leads to verbal dyspraxia",rfp,dfp)
    obj.tag_sentence("The recent spread of Corona virus flu from China to Italy,Iran, South Korea and Japan has caused global concern",rfp,dfp)
    obj.tag_sentence("Hotel California topped the singles chart",rfp,dfp)
    obj.tag_sentence("Elon Musk said Telsa will open a manufacturing plant in Europe",rfp,dfp)
    obj.tag_sentence("He flew from New York to SFO",rfp,dfp)
    obj.tag_sentence("After studies at Hofstra University , He worked for New York Telephone before He was elected to the New York State Assembly to represent the 16th District in Northwest Nassau County ",rfp,dfp)
    obj.tag_sentence("Everyday he rode his bicycle from Rajakilpakkam to Tambaram",rfp,dfp)
    obj.tag_sentence("If he loses Saturday , it could devalue his position as one of the world 's great boxers , \" Panamanian Boxing Association President Ramon     Manzanares said .",rfp,dfp)
    obj.tag_sentence("West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",rfp,dfp)
    obj.tag_sentence("they are his friends ",rfp,dfp)
    obj.tag_sentence("they flew from Boston to Rio De Janiro and had a mocha",rfp,dfp)
    obj.tag_sentence("he flew from Boston to Rio De Janiro and had a mocha",rfp,dfp)
    obj.tag_sentence("X,Y,Z are medicines",rfp,dfp)
    rfp.close()


Usage = "Usage: main_NER.py <option> [ 1 -  tag few canned sentences used in medium article.  2 - tag sentences in input file. 3 - tag single entity in a sentence\n"

if __name__== "__main__":
    if (len(sys.argv) < 2):
        print(Usage)
    else:
          obj = UnsupNER()
          if (sys.argv[1] == '1'):
              test_canned_sentences(obj)
          elif (sys.argv[1] == '2'):
              if (len(sys.argv) < 3):
                 print("Input file needs to be specified")
              else:
                 run_test(sys.argv[2],obj)
          elif (sys.argv[1] == '3'):
              if (len(sys.argv) < 3):
                 print("Input file needs to be specified")
              else:
                 tag_single_entity_in_sentence(sys.argv[2],obj)
          else:
                 print("Invalid argument:\n" + Usage)
          print("Tags and sentences are written in results.txt and debug.txt")
