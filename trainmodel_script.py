import WordEmbeddings.ComputeEmbeddings as CE
import GNN.Training as T
import CreateGraphInput as CGI
import Graph.DefineGraph as DG

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, sp_method=CE.Method.FASTTEXT, vocabulary_from_senselabeled=True)
#
DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT, slc_corpus=True)

model, train_dataloader, valid_dataloader = T.training_setup(slc_or_text_corpus=True, include_globalnode_input=False, include_sensenode_input=False, predict_senses=True, method=CE.Method.FASTTEXT, grapharea_size=32,batch_size=2, sequence_length=3);
T.training_loop(model, 0.0001, train_dataloader, valid_dataloader, 100);

# model, train_dataloader, valid_dataloader = T.training_setup(slc_or_text_corpus=False, include_globalnode_input=False, include_sensenode_input=False, predict_senses=False, method=CE.Method.FASTTEXT, grapharea_size=32,batch_size=40, sequence_length=50);
# T.training_loop(model, 0.0001, train_dataloader, valid_dataloader, 100);