import VocabularyAndEmbeddings.ComputeEmbeddings as CE;import CreateGraphInput as CGI;import Graph.DefineGraph as DG
import NN.Training as T;

CGI.exe_from_input_to_vectors(do_reset=True, compute_single_prototype=True, senselabeled_or_text=True)
#
DG.get_graph_dataobject(new=True, method=CE.Method.FASTTEXT, slc_corpus=True)

model, train_dataloader, valid_dataloader = T.setup_train(slc_or_text_corpus=True, include_globalnode_input=False,
                                                          load_saved_model=False,method=CE.Method.FASTTEXT,
                                                          grapharea_size=32, batch_size=3, sequence_length=4,
                                                          allow_dataparallel=True)
T.run_train(model, 0.0001, train_dataloader, valid_dataloader, 10, predict_senses=True, with_freezing=False)
