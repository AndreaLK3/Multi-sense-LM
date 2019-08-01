import requests
import Utils
import logging
import wiktionaryparser as WP
import re

#URL = "https://en.wiktionary.org/w/api.php"

def process_defs_subdicts(structure_ls):

    defs = []
    examples = []

    for pos_dict in structure_ls:
        defs.extend(pos_dict['text'][1:]) # skip [0] as it is grammar, not a definition
        examples.extend(pos_dict['examples'])

    return (defs, examples)


# Wiktionary definition often start with a specification, like (transitive), (obsolete), (botany), (uncountable).
# It should be removed.
def clean_definitions(defs_ls):
    new_defs = list(map(lambda definition: re.sub("^\([a-zA-Z0-9_\s,]+\)" , "", definition) , defs_ls))
    return new_defs


def get_defs_and_examples(parser, target_word):
    word_data = parser.fetch(target_word)

    defs_structure = word_data[0]['definitions']
    (raw_definitions, examples) = process_defs_subdicts(defs_structure)
    definitions = clean_definitions(raw_definitions)

    return definitions, examples


def main():
    Utils.init_logging('Wiktionary.log', logging.INFO)

    target_word = "plant"

    parser = WP.WiktionaryParser()
    definitions, _examples = get_defs_and_examples(parser, target_word)
    logging.info(definitions)


#main()