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

    #name_count_lts = list(map( lambda lm: (lm.name(), lm.count()),lemmas))
    definition = synset.definition()
    examples = synset.examples()
    #total_count = sum([tpl[1] for tpl in name_count_lts])

    return definition, examples, all_synonyms, all_antonyms


# n: POS-tagging and the different roles and meanings of a word are not addressed in this task. The purpose is to obtain
# graph-based, dictionary-enhanced word embeddings, not multi-sense
def retrieve_DESA(target_word):

    logging.info("*** WordNet : "+ target_word + ' ...')

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

