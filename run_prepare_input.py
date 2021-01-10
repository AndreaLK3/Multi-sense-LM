import VocabularyAndEmbeddings.ComputeEmbeddings as CE;import CreateGraphInput as CGI;import Graph.DefineGraph as DG
import NN.Training as T
import Graph.Adjacencies as AD
from time import time
import Utils
import logging
import os
import Filesystem as F
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Gathering data from WordNet, creating the dictionary graph')
    parser.add_argument('--grapharea_size', type=int, default=32,
                        help='number of graph nodes included in a GNN mini-batch')
    args = parser.parse_args()
    args.tied = not args.not_tied
    return args

args = parse_arguments()

Utils.init_logging('InputPipeline.log'); t0 = time()

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, senselabeled_or_text=True)
t1 = time(); Utils.time_measurement_with_msg(t0, t1, "Created vocabulary, retrieved and preprocessed input")

graph_dataobj = DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT, slc_corpus=True).to(Utils.DEVICE)
graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_SENSELABELED)
AD.get_grapharea_matrix(graph_dataobj, area_size=args.grapharea_size, hops_in_area=1,
                        graph_folder=graph_folder, new=True)
t2 = time(); Utils.time_measurement_with_msg(t1, t2, "Created dictionary graph")