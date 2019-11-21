from nltk.corpus import wordnet as wn
import Utils
import logging
import os
import re
import string
import pandas as pd

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

    definition = synset.definition()
    examples = synset.examples()

    return definition, examples, all_synonyms, all_antonyms


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
    #Utils.init_logging(os.path.join("GetKBInputData", "WordNet.log"), logging.INFO)

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



def retrieve_senses_desa(target_word):

    syns_ls = wn.synsets(target_word)
    # e.g. [Synset('sea.n.01'), Synset('ocean.n.02'), Synset('sea.n.03')]
    # note: only those synsets where the word appears first can be considered as belonging to the word.
    # Otherwise, the words is a synonym
    syns_ls = list(filter(lambda synset: target_word in synset.name(), syns_ls))

    data_lts = []

    for synset in syns_ls:
        d,e,s,a = process_synset(synset)
        data_lts.append((synset.name(),d,e,s,a))

    data_df = pd.DataFrame(data=data_lts, columns=[Utils.SENSE_WN_ID, Utils.DEFINITIONS, Utils.EXAMPLES,
                                    Utils.SYNONYMS, Utils.ANTONYMS])

    return data_df