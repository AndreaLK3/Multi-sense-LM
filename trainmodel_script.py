import CreateGraphInput as CGI
import WordEmbeddings.ComputeEmbeddings as CE
import Graph.DefineGraph as DG
import GNN.Training as T



CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, sp_method=CE.Method.FASTTEXT,
                              vocabulary_from_senselabeled=True)

DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT)

T.train(grapharea_size=32, size_batch=None, sequence_length=8, learning_rate=0.001, num_epochs=100)

