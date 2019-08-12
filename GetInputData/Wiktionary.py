import Utils
import logging
import wiktionaryparser as WP
import re

#URL = "https://en.wiktionary.org/w/api.php"

def extract_synonyms_or_antonyms(relatedWords_dict):
    nyms_lls = relatedWords_dict['words']
    nyms_lls = clear_grammar_in_brackets(nyms_lls)
    nyms_lls_clean = list(map(lambda s: s[2:], nyms_lls))  # skip ' :'
    nyms_string = ','.join(nyms_lls_clean)
    nyms = nyms_string.split(sep=',')
    return nyms


def process_defs_subdicts(structure_ls):

    defs = []
    examples = []
    synonyms = []
    antonyms = []

    for pos_dict in structure_ls:
        defs.extend(pos_dict['text'][1:]) # skip [0] as it is grammar, not a definition
        examples.extend(pos_dict['examples'])
        related_words_dictsls = pos_dict['relatedWords']
        for d in related_words_dictsls:
            if d['relationshipType'] == 'synonyms':
                synonyms = extract_synonyms_or_antonyms(d)
            if d['relationshipType'] == 'antonyms':
                antonyms = extract_synonyms_or_antonyms(d)


    return (defs, examples, synonyms, antonyms)


# Wiktionary definition often start with a specification, like (transitive), (obsolete), (botany), (uncountable).
# It should be removed.
def clear_grammar_in_brackets(defs_ls):
    new_defs = list(map(lambda definition: re.sub("^\([a-zA-Z0-9_\s,]+\)" , "", definition) , defs_ls))
    return new_defs


def retrieve_DESA(target_word):
    parser = WP.WiktionaryParser()
    word_data = parser.fetch(target_word)

    defs_structure = word_data[0]['definitions']
    (raw_definitions, examples, synonyms, antonyms) = process_defs_subdicts(defs_structure)
    definitions = clear_grammar_in_brackets(raw_definitions)

    return definitions, examples, synonyms, antonyms


def main():
    Utils.init_logging('Wiktionary.log', logging.INFO)

    target_word = "plant"

    definitions, _examples, _synonyms, _antonyms = retrieve_DESA(target_word)
    logging.info(definitions)


#main()