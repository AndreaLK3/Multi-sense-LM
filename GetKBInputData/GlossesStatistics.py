import Utils
import Filesystem as F
import os
import pandas as pd
from math import inf
import nltk
import re
import string
import logging

# Highlight the minimum, average and maximum length, in words, of all the definitions / examples we
# retrieved from WordNet, in raw and processed form both
def count_glosses_length(glosses_id=0):
    if glosses_id==0:
        elems_name=Utils.DEFINITIONS
    elif glosses_id==1:
        elems_name=Utils.EXAMPLES
    else:
        raise Exception("glosses_id must be either 0 (definitions) or 1 (examples)")

    Utils.init_logging("GlossesStatistics_"+elems_name+".log")
    input_folder_fpath = os.path.join(F.FOLDER_INPUT, F.FOLDER_SENSELABELED)
    # process== eliminating duplicate defs & examples. Did not modify the sentences
    processed_elems_fpath = os.path.join(input_folder_fpath, Utils.PROCESSED + '_' + elems_name + ".h5")
    # example of processed_definition:  report.n.01 | a written document describing the findings of some individual or group

    elems_df = pd.read_hdf(processed_elems_fpath)
    elems_ls = elems_df[elems_name].to_list()

    min_len = inf
    max_len = 0
    sum_len = 0
    count_elems = 0

    for elem_txt in elems_ls:
        elem_txt_nopunct = re.sub("["+str(string.punctuation)+"]", " ", elem_txt)
        elem_len = len(elem_txt_nopunct.split())
        if elem_len==0:
            logging.info("Gloss of length=0 found at element n." + str(count_elems));
            continue
        if elem_len < min_len:
            min_len = elem_len
        if elem_len > max_len:
            max_len = elem_len
        sum_len = sum_len + elem_len
        count_elems = count_elems + 1
        if (count_elems+1) % 5000 == 0:
            logging.info("Examined gloss:" + elems_name + " n."+str(count_elems))

    avg_len = sum_len / count_elems
    logging.info("Statistics for the glosses:" + elems_name + ":")
    logging.info("Minimum length = " + str(min_len) + " ; maximum length = " + str(max_len) +
                 " average length = " + str(avg_len))
