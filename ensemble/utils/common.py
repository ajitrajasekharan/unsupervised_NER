import pdb
import sys

WORD_POS = 1
TAG_POS = 2
MASK_TAG = "__entity__"
INPUT_MASK_TAG = ":__entity__"
RESET_POS_TAG='RESET'


noun_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','POS','CD']
cap_tags = ['NFP','JJ','NN','FW','NNS','NNPS','JJS','JJR','NNP','PRP']


def generate_masked_sentences(terms_arr):
    size = len(terms_arr)
    sentence_arr = []
    span_arr = []
    i = 0
    while (i < size):
        term_info = terms_arr[i]
        if (term_info[TAG_POS] in noun_tags):
            skip = gen_sentence(sentence_arr,terms_arr,i)
            i +=  skip
            for j in range(skip):
                span_arr.append(1)
        else:
            i += 1
            span_arr.append(0)
    #print(sentence_arr)
    return sentence_arr,span_arr


def gen_sentence(sentence_arr,terms_arr,index):
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



def capitalize(terms_arr):
    for i,term_tag in enumerate(terms_arr):
        #print(term_tag)
        if (term_tag[TAG_POS] in cap_tags):
            word = term_tag[WORD_POS][0].upper() + term_tag[WORD_POS][1:]
            term_tag[WORD_POS] = word
    #print(terms_arr)

def set_POS_based_on_entities(sent):
    terms_arr = []
    sent_arr = sent.split()
    for i,word in enumerate(sent_arr):
        #print(term_tag)
        term_tag = ['-']*5
        if (word.endswith(INPUT_MASK_TAG)):
            term_tag[TAG_POS] = noun_tags[0]
            term_tag[WORD_POS] = word.replace(INPUT_MASK_TAG,"")
        else:
            term_tag[TAG_POS] = RESET_POS_TAG
            term_tag[WORD_POS] = word
        terms_arr.append(term_tag)
    return terms_arr
    #print(terms_arr)

def filter_common_noun_spans(span_arr,masked_sent_arr,terms_arr,common_descs):
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
                if (term[WORD_POS].lower() not in common_descs):
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

def normalize_casing(sent):
    sent_arr = sent.split()
    ret_sent_arr = []
    for i,word in enumerate(sent_arr):
        if (len(word) > 1):
            norm_word = word[0] + word[1:].lower()
        else:
            norm_word = word[0]
        ret_sent_arr.append(norm_word)
    return ' '.join(ret_sent_arr)

