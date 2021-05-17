from InputPipeline import CreateGraphInput as CGI
import Graph.DefineGraph as DG
import Graph.Adjacencies as AD
from time import time
import Utils
import Filesystem as F
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Gathering data from WordNet, creating the dictionary graph')
    parser.add_argument('--grapharea_size', type=int, default=32,
                        help='number of graph nodes included in a GNN mini-batch')
    # WikiText-103 is not used as a vocabulary source / pre-training corpus for StandardLM in the final version
    # parser.add_argument('--txt_corpus', type=str,  choices=['wt2', 'wt103'], default="wt2",
    #                    help='Whether to run the pipeline with a vocabulary from WT2+SemCor or from WT103. ' \
    #                         'Using the former alternative if unspecified')
    args = parser.parse_args()
    return args

args = parse_arguments()

Utils.init_logging('InputPipeline.log'); t0 = time()
vocabulary_sources_ls = [F.WT2, F.SEMCOR] # [F.WT103] if args.txt_corpus=="wt103" else

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, vocabulary_sources_ls=vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT)
t1 = time(); Utils.time_measurement_with_msg(t0, t1, "Created vocabulary, retrieved and preprocessed input")

graph_dataobj = DG.get_graph_dataobject(new=True, vocabulary_sources_ls=vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT).to(Utils.DEVICE)
graph_folder, _, _ = F.get_folders_graph_input_vocabulary(vocabulary_sources_ls, sp_method=Utils.SpMethod.FASTTEXT)
AD.get_grapharea_matrix(graph_dataobj, area_size=args.grapharea_size, hops_in_area=1, graph_folder=graph_folder, new=True)
t2 = time(); Utils.time_measurement_with_msg(t1, t2, "Created dictionary graph")