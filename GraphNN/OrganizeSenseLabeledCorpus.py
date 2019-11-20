import os
import Filesystem as F
import logging
import Utils
import lxml.etree
import pandas as pd
import time

def load_NOAD_to_WordNet_mapping():
    manualmap_df = pd.read_csv(os.path.join(F.FOLDER_TEXT_CORPUSES, F.NOAD_WORDNET_MANUALMAP_FILE),
                               names=[Utils.SENSE_NOAD,Utils.SENSE_WORDNET], sep='\t')
    algorithmicmap_df = pd.read_csv(os.path.join(F.FOLDER_TEXT_CORPUSES, F.NOAD_WORDNET_AUTOMAP_FILE),
                               names=[Utils.SENSE_NOAD,Utils.SENSE_WORDNET], sep='\t')
    manually_covered_noad_senses_ls = manualmap_df.loc[:,Utils.SENSE_NOAD].to_list()
    map_rows_toadd_ls = []
    duplicates_count = 0
    for row_tpl in algorithmicmap_df.itertuples():
        if not(getattr(row_tpl, Utils.SENSE_NOAD) in manually_covered_noad_senses_ls):
            map_rows_toadd_ls.append(row_tpl)
        else:
            duplicates_count = duplicates_count + 1
            logging.debug("NOAD sense (" + str(getattr(row_tpl, Utils.SENSE_NOAD)) +") already manually mapped. Skipping")
    logging.info(str(duplicates_count)+" senses from NOAD were already manually mapped to WordNet senses...")

    toadd_df = pd.DataFrame(map_rows_toadd_ls)
    toadd_df_01 = toadd_df.drop('Index', axis='columns')

    final_mapping_df = pd.concat([manualmap_df,toadd_df_01])

    return final_mapping_df


def get_WordNet_sense(noad_sense, mapping_df):
    try:
        wn_sense = mapping_df.loc[mapping_df[Utils.SENSE_NOAD] == noad_sense].values[0][1]
        logging.debug(wn_sense)
    except IndexError:
        logging.warning("A Sense from NOAD: " + str(noad_sense) + " was not found in the mapping. Skipping")
        wn_sense = Utils.EMPTY
    return wn_sense


def get_breakspace_token(token_code):
    if token_code == 'SPACE_BREAK':
        return ' '
    elif token_code == 'SENTENCE_BREAK':
        return '\n'


def process_word(word_elem, sense_mapping_df):

    attributes_dict = word_elem.attrib
    keys = attributes_dict.keys()
    preceding_break = attributes_dict['break_level']
    token = attributes_dict['text']
    if 'sense' in keys:
        noad_sense = attributes_dict['sense']
        wn_sense = get_WordNet_sense(noad_sense, sense_mapping_df)
        lemma = attributes_dict['lemma']
        pos = attributes_dict['pos']
    else:
        wn_sense = Utils.EMPTY
        lemma = Utils.EMPTY
        pos = Utils.EMPTY
    # if we have NO_BREAK, do not add anything
    if token == 'NO_BREAK':
        return
    # Otherwise:
    # first, write in the corpus archive the preceding break (space, tab, newline)
    break_data_tpl = (get_breakspace_token(preceding_break), Utils.EMPTY, Utils.EMPTY, Utils.EMPTY)
    # then, write the word (and its sense & co., if present)
    token_data_tpl = (token, lemma, pos, wn_sense)


    return break_data_tpl, token_data_tpl


def process_xml_document(xml_fpath, out_archive, sense_mapping_df):
    h5_itemsizes = {'token': Utils.HDF5_BASE_SIZE_512, 'lemma': Utils.HDF5_BASE_SIZE_512 / 2,
                    'pos': Utils.HDF5_BASE_SIZE_512 / 16, Utils.SENSE_WORDNET: Utils.HDF5_BASE_SIZE_512}

    xml_docfile = open(xml_fpath, "rb")
    data_tpl_ls = []

    for event, elem in lxml.etree.iterparse(xml_docfile):
        if elem.tag == "word":
            break_data_tpl, token_data_tpl = process_word(elem, sense_mapping_df)
            data_tpl_ls.append(break_data_tpl)
            data_tpl_ls.append(token_data_tpl)

    new_df = pd.DataFrame(data=data_tpl_ls, columns=['token', 'lemma', 'pos', Utils.SENSE_WORDNET])
    out_archive.append(key='SenseLabeledCorpus', value=new_df, min_itemsize=h5_itemsizes)



def process_corpus_subfiles(source_filepaths, destination_archive, sense_mapping_df):

    for src_fpath in source_filepaths:
        logging.debug(src_fpath)
        logging.debug(os.listdir(src_fpath))
        sub_xml_filenames = list(filter(lambda fname: 'xml' in fname, os.listdir(src_fpath)))
        sub_xml_filepaths = list(map(lambda fn: os.path.join(src_fpath, fn), sub_xml_filenames))
        for sub_xml_filepath in sub_xml_filepaths:
            logging.info("Processing the sense-labeled document located at: " + str(sub_xml_filepath) + " ...")
            process_xml_document(sub_xml_filepath, destination_archive, sense_mapping_df)


def exe_semcore_and_masc():
    Utils.init_logging('temp.log')

    # --resetting from last time
    out_archive_semcor = pd.HDFStore(os.path.join(F.FOLDER_INPUT, F.SEMCOR_H5_FILE), mode='w')
    out_archive_masc =pd.HDFStore(os.path.join(F.FOLDER_INPUT, F.MASC_H5_FILE), mode='w')

    semcor_source_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_SEMCOR)
    masc_base_fpath = os.path.join(F.FOLDER_TEXT_CORPUSES, F.FOLDER_MASC, F.FOLDER_MASC_WRITTEN)
    masc_source_fpaths = list(map(lambda dir_tpl: dir_tpl[0], os.walk(masc_base_fpath)))

    noad_to_Wordnet_mapping = load_NOAD_to_WordNet_mapping()

    process_corpus_subfiles([semcor_source_fpath], out_archive_semcor, noad_to_Wordnet_mapping)
    process_corpus_subfiles(masc_source_fpaths, out_archive_masc, noad_to_Wordnet_mapping)

    out_archive_semcor.close()
    out_archive_masc.close()


