import VocabularyAndEmbeddings.ComputeEmbeddings as CE;import CreateGraphInput as CGI;import Graph.DefineGraph as DG
import NN.Training as T;

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, senselabeled_or_text=True)
#
DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT, slc_corpus=True)

