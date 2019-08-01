from nltk.corpus import wordnet as wn
import Utils
import logging


# We examine here one synset, that was found by searching for the target word.
# Objectives:
# - Definitions, to use for the target word.
# - Possibly, the frequency count and word position in the lemmas,
#   to distinguish between relevant and distant/synonym definitions.  To do in version > 1.0
def process_synset(synset):
    lemmas = synset.lemmas()

    #name_count_lts = list(map( lambda lm: (lm.name(), lm.count()),lemmas))
    definition_WN = synset.definition()
    #total_count = sum([tpl[1] for tpl in name_count_lts])

    return definition_WN


# n: POS-tagging and the different roles and meanings of a word are not addressed in this task. The purpose is to obtain
# graph-based, dictionary-enhanced word embeddings, not multi-sense
def process_all_synsets_of_word(target_word =):
    Utils.init_logging("defs_WordNet.log", logging.INFO)

    defs_WN = []
    syns_ls = wn.synsets(target_word)
    logging.info(syns_ls)

    for syn in syns_ls:
        defs_WN.append(process_synset(syn))

    return defs_WN

#defs_WN = process_all_synsets_of_word('plant')
#logging.info("\n".join(defs_WN))