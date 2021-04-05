import logging
import os
from types import SimpleNamespace

import pandas as pd

import Filesystem as F
import Utils
from Graph.Adjacencies import get_node_data, lemmatize_node


# Utility function: determine which globals have more than 1 sense, versus the dummySenses and 0or1 sense.
# (We evaluate the number of senses of the lemmatized form). Used to compute different Perplexities
def compute_globals_numsenses(graph_dataobj, grapharea_matrix, grapharea_size, inputdata_folder, vocab_fpath):
    num_senses_ls = []
    last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    last_idx_globals = graph_dataobj.node_types.tolist().index(2)

    first_idx_dummySenses = Utils.get_startpoint_dummySenses(inputdata_folder)
    logging.info("Examining the graph + corpus, to determine which globals have multiple senses...")

    vocabulary_df = pd.read_hdf(vocab_fpath)
    vocabulary_wordList = vocabulary_df['word'].to_list().copy()
    vocabulary_lemmatizedWordsList = vocabulary_df['lemmatized_form'].to_list().copy()
    vocabulary_frequencyList = vocabulary_df['frequency'].to_list().copy()

    log_idx = 10000
    # iterate over the globals
    for idx in range(last_idx_senses, last_idx_globals):
        if idx % log_idx == 0: logging.info("global idx=" + str(idx) + "...")
        adjacent_nodes, edges, edge_type = get_node_data(grapharea_matrix, idx, grapharea_size, features_mask=(True, True, True))

        args_dict = {'first_idx_dummySenses': first_idx_dummySenses, 'last_idx_senses':last_idx_senses,
                     'vocabulary_wordList':vocabulary_wordList, 'vocabulary_lemmatizedList':vocabulary_lemmatizedWordsList,
                     'grapharea_matrix':grapharea_matrix, 'grapharea_size':grapharea_size}  # packing the parameters for the lemmatizer function
        args = SimpleNamespace(**args_dict)
        adjacent_nodes, edges, edge_type = lemmatize_node(adjacent_nodes, edges, edge_type, args)

        num_senses = edge_type.tolist().count(2)
        num_dummy_senses = len(list(filter(lambda n: first_idx_dummySenses < n and n < last_idx_senses, adjacent_nodes)))
        num_senses = num_senses - num_dummy_senses

        logging.debug("word=" + str(vocabulary_wordList[idx-last_idx_senses]) +
                         " ; edge_type="+str(edge_type) + " ; num_senses=" + str(num_senses))
        num_senses_ls.append(num_senses)

    new_vocabulary_data = list(zip(vocabulary_wordList, vocabulary_frequencyList, vocabulary_lemmatizedWordsList, num_senses_ls))
    new_vocabulary_df = pd.DataFrame(data=new_vocabulary_data, columns=['word','frequency','lemmatized_form','num_senses'])
    vocabulary_h5_archive = pd.HDFStore(vocab_fpath, mode='w')
    min_itemsize_dict = {'word': Utils.HDF5_BASE_SIZE_512 / 4, 'frequency': Utils.HDF5_BASE_SIZE_512 / 8,
                         'lemmatized_form': Utils.HDF5_BASE_SIZE_512 / 4, 'num_senses': Utils.HDF5_BASE_SIZE_512 / 8}
    vocabulary_h5_archive.append(key='vocabulary', value=new_vocabulary_df, min_itemsize=min_itemsize_dict)
    vocabulary_h5_archive.close()
    return new_vocabulary_df


def get_polysenseglobals_dict(vocab_sources_ls, sp_method, thresholds=(2,3,5,10,30)):
    _, _, vocab_folder = F.get_folders_graph_input_vocabulary(vocab_sources_ls, sp_method)
    vocab_fpath = os.path.join(vocab_folder, "vocabulary.h5")
    vocabulary_df = pd.read_hdf(vocab_fpath)
    vocabulary_num_senses_ls = vocabulary_df['num_senses'].to_list().copy()

    numsenses_wordindices_dict = {}.fromkeys(thresholds)

    for threshold_key in thresholds:
        numsenses_wordindices_dict[threshold_key] = set()
        for i in range(len(vocabulary_num_senses_ls)):
            if vocabulary_num_senses_ls[i] >= threshold_key:
                numsenses_wordindices_dict[threshold_key].add(i)

    return numsenses_wordindices_dict