import VocabularyAndEmbeddings.ComputeEmbeddings as CE;import CreateGraphInput as CGI;import Graph.DefineGraph as DG
import NN.Training as T
import Graph.Adjacencies as AD
from time import time
import Utils
import logging
import os
import Filesystem as F

Utils.init_logging('Pipeline_SLC.log')
t0 = time()

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, senselabeled_or_text=True)
t1 = time()
logging.info("Created vocabulary, retrieved and preprocessed input. Time elapsed=" + str(t1 - t0))


graph_dataobj = DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT, slc_corpus=True).to(Utils.DEVICE)
graph_folder = os.path.join(F.FOLDER_GRAPH, F.FOLDER_SENSELABELED)
AD.get_grapharea_matrix(graph_dataobj, area_size=32, hops_in_area=1, graph_folder=graph_folder, new=True)
t2 = time()
logging.info("Created dictionary graph. Time elapsed=" + str(round(t2 - t1,3)))

model, datasets, dataloaders = T.setup_train(slc_or_text_corpus=True, model_type=T.ModelType.SELECTK, K=1, include_globalnode_input=0)
T.run_train(model, dataloaders, 0.00005, 24)
t3 = time()
logging.info("Trained model. Time elapsed=" + str(round(t3-t2,3)))