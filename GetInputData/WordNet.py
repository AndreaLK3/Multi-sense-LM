from nltk.corpus import wordnet as wn
import Utils
import logging
import os
import re
import string

# We examine here one synset, that was found by searching for the target word.
# Objectives:
# - Definitions; examples; synonyms (from the lemmas in the same synset)
# - Possibly, the frequency count and word position in the lemmas,
#   to distinguish between relevant and distant/synonym definitions.  To do in version > 1.0
def process_synset(synset):
    lemmas = synset.lemmas()

    all_synonyms = list(map( lambda l: l.name(),lemmas))

    all_antonyms = []
    for l in lemmas:
        if l.antonyms():
            antonyms_to_add = list(map(lambda a_lemma: a_lemma.name(),l.antonyms()))
            all_antonyms.extend(antonyms_to_add)

    #definition = synset.definition()
    # examples = synset.examples()

    return all_synonyms, all_antonyms


def lookup_bndefs_dictionary(wn_def, bn_defs_dict):

    for key_bn_id in bn_defs_dict.keys():
        bn_def_ls = list(map(lambda d: re.sub('['+string.punctuation+']',"",d.lower()), bn_defs_dict[key_bn_id]))
        wn_def_01 = re.sub('['+string.punctuation+']',"",wn_def.lower())
        if wn_def_01 in bn_def_ls:
            logging.debug("Match found for wn_def='" + str(wn_def) +"' at bn_id=" + str(key_bn_id))
            return key_bn_id
        else:
            logging.debug("Match not found between wn_def='" + str(wn_def) +"' in bn_def_ls=" + str(bn_def_ls))
            continue
    return None



def retrieve_SA_bySenses(target_word, bn_defs_dict):
    #Utils.init_logging(os.path.join("GetInputData", "WordNet.log"), logging.INFO)

    synonyms_dict = {}
    antonyms_dict = {}

    syns_ls = wn.synsets(target_word)

    for synset in syns_ls:
        wn_def = synset.definition()
        bn_id = lookup_bndefs_dictionary(wn_def, bn_defs_dict)
        if bn_id is not None:
            synonyms, antonyms = process_synset(synset)
            synonyms_dict[bn_id] = synonyms
            logging.debug("From WordNet: synonyms : " + str(synonyms))
            antonyms_dict[bn_id] = antonyms

    return synonyms_dict, antonyms_dict