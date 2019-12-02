import os
import Filesystem as F
import logging
import Utils
import lxml.etree
import pandas as pd
import sqlite3


def get_breakspace_token(token_code):
    if token_code == 'SPACE_BREAK':
        return ' '
    elif token_code == 'SENTENCE_BREAK':
        return '\n'


# read an XML in UFSAC sense-labeled format. __next__() returns
def dataset_generator(xml_fpath):
    xml_docfile = open(xml_fpath, "rb")

    for event, elem in lxml.etree.iterparse(xml_docfile):
        if elem.tag == "sentence":
            yield({'surface_form':Utils.EOS_TOKEN}) # following the format of attribute dictionaries of <word> elements
        if elem.tag == "word":
            yield(elem.attrib)



def exe():
    Utils.init_logging('temp.log')
    xml_fnames = ['semcor.xml', 'masc.xml', 'omsti.xml', 'raganato_ALL.xml', 'wngt.xml']
    xml_fpaths = list(map(
        lambda fname: os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_SENSEANNOTATED, F.FOLDER_UFSAC, fname), xml_fnames)

    for token_dict in dataset_generator(xml_fpath):
        logging.info(token_dict)