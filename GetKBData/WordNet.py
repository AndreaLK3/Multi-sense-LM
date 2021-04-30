from nltk.corpus import wordnet as wn

import Lexicon
import Utils
import logging
import os
import re
import string
import pandas as pd

# We examine here one synset, that was found by searching for the target word.
# Objectives:
# - Definitions; examples; synonyms (from the lemmas in the same synset, must exclude the target word itself)
# - Possibly, the frequency count and word position in the lemmas,
#   to distinguish between relevant and distant/synonym definitions.  To do in version > 1.0
def process_synset(target_word, synset):
    lemmas = synset.lemmas()

    all_synonyms = list(map(lambda l: l.name(),lemmas))
    all_synonyms = list(filter(lambda l: l != target_word, all_synonyms))

    all_antonyms = []
    for l in lemmas:
        if l.antonyms():
            antonyms_to_add = list(map(lambda a_lemma: a_lemma.name(),l.antonyms()))
            all_antonyms.extend(antonyms_to_add)

    definition = synset.definition()
    examples = synset.examples()

    return definition, examples, all_synonyms, all_antonyms


def retrieve_senses_desa(target_word):

    syns_ls = wn.synsets(target_word)
    # [Synset('bank.n.01'), Synset('depository_financial_institution.n.01'), Synset('bank.n.03'), Synset('bank.n.04'),
    # Synset('bank.n.05'), Synset('bank.n.06'), Synset('bank.n.07'), Synset('savings_bank.n.02'), Synset('bank.n.09'),
    # Synset('bank.n.10'), Synset('bank.v.01'), Synset('bank.v.02'), Synset('bank.v.03'), Synset('bank.v.04'),
    # Synset('bank.v.05'), Synset('deposit.v.02'), Synset('bank.v.07'), Synset('trust.v.01')]
    # note: only those synsets where the word appears first can be considered as belonging to the word.
    # Otherwise, the word is a synonym
    syns_ls = list(filter(
        lambda synset: target_word.lower() == (Utils.get_word_from_sense(synset.name())).lower(), syns_ls))
    logging.debug("WordNet.retrieve_senses_desa(target_word) > " + " synsets where the target_word word appears first " +
                 "syns_ls=" + str(syns_ls))

    data_lts = []

    for synset in syns_ls:
        d,e,s,a = process_synset(target_word, synset)
        logging.debug("WordNet.retrieve_senses_desa(target_word) > " + " for synset=" + synset.name() +
                     " we retrieved the definition, " + str(len(e)) + " examples, " + str(len(s)) + " synonyms and " +
                     str(len(a)) + " antonyms")
        data_lts.append((synset.name(),d,e,s,a))

    data_df = pd.DataFrame(data=data_lts, columns=[Lexicon.SENSE_WN_ID, Lexicon.DEFINITIONS, Lexicon.EXAMPLES,
                                                   Lexicon.SYNONYMS, Lexicon.ANTONYMS])

    return data_df