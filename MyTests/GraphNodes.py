import Graph.DefineGraph as DG
import Graph.Adjacencies as AD
import Utils
import random
import logging
import os
import Filesystem as F
import pandas as pd
from NN.ExplorePredictions import get_sense_fromindex, get_globalword_fromindex_df

def log_node(n, last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df):
    last_idx_senses , last_idx_globals, last_idx_definitions = last_indices_tpl
    if n < last_idx_senses:
        sense = get_sense_fromindex(n, inputdata_folder)
        logging.info("n=" + str(n) + " ; sense=" + str(sense))
    elif last_idx_senses < n and n < last_idx_globals:
        idx = n - last_idx_senses
        global_word = get_globalword_fromindex_df(idx, vocabulary_folder)
        logging.info("n=" + str(n) + " ; global=" + str(global_word))
    elif last_idx_globals < n and n < last_idx_definitions:
        idx = (n - last_idx_globals)
        logging.info("definition: n=" + str(n) + str((definitions_df.iloc[idx].sense_wn_id, definitions_df.iloc[idx].definitions)))
    elif last_idx_definitions < n:
        idx = (n - last_idx_definitions)
        logging.info("example: n=" + str(n) + str((examples_df.iloc[idx].sense_wn_id, examples_df.iloc[idx].examples)))


def test(slc_or_text):
    Utils.init_logging("Test-GraphNodes.log")

    logging.info("Graph test. On: SenseLabeled corpus")
    graph_dataobj = DG.get_graph_dataobject(new=False, slc_corpus=slc_or_text)

    subfolder = F.FOLDER_SENSELABELED if slc_or_text else F.FOLDER_STANDARDTEXT
    graph_folder_slc = os.path.join(F.FOLDER_GRAPH, subfolder)
    vocabulary_folder = os.path.join(F.FOLDER_VOCABULARY, subfolder)
    inputdata_folder = os.path.join(F.FOLDER_INPUT, subfolder)

    definitions_h5 = os.path.join(inputdata_folder, Utils.PROCESSED + '_' + Utils.DEFINITIONS + ".h5")
    definitions_df = pd.read_hdf(definitions_h5, key=Utils.DEFINITIONS, mode="r")
    examples_h5 = os.path.join(inputdata_folder, Utils.PROCESSED + '_' + Utils.EXAMPLES + ".h5")
    examples_df = pd.read_hdf(examples_h5, key=Utils.EXAMPLES, mode="r")

    grapharea_matrix = AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder_slc)

    last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    last_idx_globals = graph_dataobj.node_types.tolist().index(2)
    last_idx_definitions = graph_dataobj.node_types.tolist().index(3)
    num_nodes = len(graph_dataobj.node_types.tolist())
    last_indices_tpl = (last_idx_senses , last_idx_globals, last_idx_definitions)

    random_nodes = [random.randint(0, last_idx_senses) for _i in range(10)] + \
                   [random.randint(last_idx_senses, last_idx_globals) for _i in range(10)] + \
                   [random.randint(last_idx_globals, last_idx_definitions) for _i in range(5)] + \
                   [random.randint(last_idx_definitions, num_nodes) for _i in range(5)]

    for center in random_nodes:
        logging.info("\nCenter:")
        log_node(center, last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df)

        neighbours, _, _ = AD.get_node_data(grapharea_matrix, center, grapharea_size=32, features_mask=(True, False, False))
        logging.info("Neighbours:")
        for n_t in neighbours:
            log_node(n_t.item(), last_indices_tpl, inputdata_folder, vocabulary_folder, definitions_df, examples_df)
