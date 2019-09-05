from nltk.corpus import wordnet as wn
import Utils
import logging


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
        ants = l.antonyms()
        for a in ants:
            all_antonyms.append(a.name())

    #definition = synset.definition()
    examples = synset.examples()

    return examples, all_synonyms, all_antonyms


# n: POS-tagging and the different roles and meanings of a word are not addressed in this task. The purpose is to obtain
# graph-based, dictionary-enhanced word embeddings, not multi-sense
def retrieve_DESA(target_word):

    defs = []
    examples = []
    synonyms = []
    antonyms = []
    syns_ls = wn.synsets(target_word)
    logging.debug(syns_ls)

    for syn in syns_ls:
        def_exs_syns_ants = process_synset(syn)
        defs.append(def_exs_syns_ants[0])
        examples.extend(def_exs_syns_ants[1])
        synonyms.extend(list(filter (lambda s: s != target_word, def_exs_syns_ants[2])) )
        antonyms.extend(list(filter(lambda s: s != target_word, def_exs_syns_ants[3])))

    #eliminate duplicates coming from different senses and synsets
    synonyms = list(dict.fromkeys(synonyms))
    antonyms = list(dict.fromkeys(antonyms))

    return (defs, examples, synonyms, antonyms)



def lookup_bndefs_dictionary(wn_def, bn_defs_dict):

    for key_bn_id in bn_defs_dict.keys():
        bn_def_ls = bn_defs_dict[key_bn_id]
        if wn_def in bn_def_ls:
            logging.info("Match found for def='" + str(wn_def) +"' at bn_id=" + str(key_bn_id))
            return key_bn_id
        else:
            continue
    return None



def retrieve_ESA_bySenses(target_word, bn_defs_dict):

    examples_dict = {}
    synonyms_dict = {}
    antonyms_dict = {}

    syns_ls = wn.synsets(target_word)

    for synset in syns_ls:
        wn_def = synset.definition()
        bn_id = lookup_bndefs_dictionary(wn_def, bn_defs_dict)
        if bn_id is not None:
            examples, synonyms, antonyms = process_synset(synset)
            examples_dict[bn_id] = examples
            synonyms_dict[bn_id] = synonyms
            antonyms_dict[bn_id] = antonyms

    return examples_dict, synonyms_dict, antonyms_dict