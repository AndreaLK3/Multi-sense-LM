from enum import Enum
import torch
import Models.Variants.InputSignals as InputSignals
import Utils
from torch.nn.parameter import Parameter


##### Initialization of: graph_dataobj, grapharea_matrix, vocabulary_lists & more
def init_model_parameters(model, graph_dataobj, grapharea_size, grapharea_matrix,
                          vocabulary_df, batch_size, n_layers, hidden_units):
    model.grapharea_matrix = grapharea_matrix
    model.grapharea_size = grapharea_size

    model.vocabulary_df = vocabulary_df
    model.vocabulary_wordList = vocabulary_df['word'].to_list().copy()
    model.vocabulary_lemmatizedList = vocabulary_df['lemmatized_form'].to_list().copy()

    model.predict_senses = False # it can be set to True when starting a training loop

    model.first_idx_dummySenses = Utils.compute_startpoint_dummySenses(graph_dataobj) # used to lemmatizeNode in GNNs
    model.last_idx_senses = graph_dataobj.node_types.tolist().index(1)
    model.last_idx_globals = graph_dataobj.node_types.tolist().index(2)

    # These are used for the Senses' GRU
    model.batch_size = batch_size
    model.n_layers = n_layers
    model.hidden_size = hidden_units
    # utility for select_index
    model.select_first_indices = Parameter(torch.tensor(list(range(2 * hidden_units))).to(torch.float32),
                                          requires_grad=False)
    return

##### Choose the method to compute the location context / Self-attention query
class ContextMethod(Enum):
    AVERAGE = "The context at location t is the average of the last [t-C,...,t] tokens"
    GRU = "The context at location t is the output of a 2-layer GRU"


##### Assign =1 to a specific global or sense in the probability distribution #####
def assign_one(target_elements, seq_len, distributed_batch_size, len_output_distr, current_device):

    # ----- preparing the base for the artificial softmax -----
    softmax_distribution = torch.ones((seq_len, distributed_batch_size, len_output_distr)).to(current_device)
    epsilon = 10 ** (-8)
    softmax_distribution = epsilon * softmax_distribution  # base probability value for non-selected senses
    mask = torch.zeros(size=(seq_len, distributed_batch_size, len_output_distr)).to(torch.bool).to(current_device)

    targets = target_elements.clone().reshape((seq_len, distributed_batch_size)).to(torch.torch.int64).to(
        current_device)
    # ----- writing in the artificial softmax -----
    for t in (range(seq_len)):
        for b in range(distributed_batch_size):
            mask[t, b, targets[t, b]] = True
    assign_one = (torch.ones(
        size=mask[mask == True].shape)).to(current_device)
    softmax_distribution.masked_scatter_(mask=mask.data.clone(), source=assign_one)

    predictions = torch.log(softmax_distribution).reshape(seq_len * distributed_batch_size,
                                                           softmax_distribution.shape[2])
    return predictions


# Alternative to GRUs: a Transformer-XL architecture
def predict_withTXL(transformer, mems, input_indices):

    output_obj = transformer(input_ids=input_indices, mems=mems)
    # output_obj = model.standard_lm_transformer(inputs_embeds=batch_input_signals)
    probabilities = output_obj.prediction_scores
    predictions = torch.reshape(probabilities,
                                shape=(probabilities.shape[0] * probabilities.shape[1], probabilities.shape[2]))  # 48, 1, 44041
    return predictions, mems

# The first part of the forward() call common to all model variants. Extracts the input signals from the batchinput_tensor,
# which can also be used in the senses' architecture, and predict the next word
def get_input_and_predict_globals(model, batchinput_tensor, batch_labels):
    # -------------------- Init --------------------
    if batchinput_tensor.shape[1] > 1:
        time_instants = torch.chunk(batchinput_tensor, chunks=batchinput_tensor.shape[1], dim=1)
    else:
        time_instants = [batchinput_tensor]

    word_embeddings_ls = []
    currentglobal_nodestates_ls = []
    globals_input_ids_ls = []  # for the transformer

    # -------------------- Compute and collect input signals; predict globals -------------------
    for batch_elements_at_t in time_instants:
        InputSignals.get_input_signals(model, batch_elements_at_t, word_embeddings_ls, currentglobal_nodestates_ls, globals_input_ids_ls)

    word_embeddings = torch.stack(word_embeddings_ls, dim=0)
    global_nodestates = torch.stack(currentglobal_nodestates_ls,
                                    dim=0) if model.StandardLM.include_globalnode_input > 0 else None
    batch_input_signals_ls = list(filter(lambda signal: signal is not None,
                                         [word_embeddings, global_nodestates]))
    batch_input_signals = torch.cat(batch_input_signals_ls, dim=2)
    predictions_globals, _ = model.StandardLM(batchinput_tensor, batch_labels)

    return batch_input_signals, globals_input_ids_ls, word_embeddings, predictions_globals