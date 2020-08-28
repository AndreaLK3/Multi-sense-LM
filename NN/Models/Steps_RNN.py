import torch
from torch.nn import Parameter, functional as tfunc

#############################
###### RNN operations #######
#############################

# The loop over the layers of a RNN
def rnn_loop(batch_input_signals, model):
    main_rnn_out = None
    input = batch_input_signals
    for i in range(model.n_layers):
        layer_rnn = model.main_rnn_ls[i]
        layer_rnn.flatten_parameters()
        if model.model_type.upper() == "LSTM":
            main_rnn_out, (hidden_i, cells_i) = \
                layer_rnn(input, select_layer_memory(model, i, layer_rnn))
            update_layer_memory(model, i, layer_rnn, hidden_i, cells_i)
        else: # GRU
            main_rnn_out, hidden_i = \
                layer_rnn(input, select_layer_memory(model, i, layer_rnn))
            update_layer_memory(model, i, layer_rnn, hidden_i)

        input = main_rnn_out

    return main_rnn_out


# Reshaping the hidden state memories when we know the batch size allocated on the current GPU
def reshape_memories(distributed_batch_size, model):

    new_num_hidden_state_elems = model.n_layers * distributed_batch_size * model.hidden_size
    model.memory_hn = Parameter(torch.reshape(model.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                              (model.n_layers, distributed_batch_size, model.hidden_size)),
                                requires_grad=False)
    model.memory_cn = Parameter(torch.reshape(model.memory_cn.flatten()[0:new_num_hidden_state_elems],
                                              (model.n_layers, distributed_batch_size, model.hidden_size)),
                                requires_grad=False)
    model.memory_hn_senses = Parameter(
        torch.reshape(
            model.memory_hn_senses.flatten()[0:((model.n_layers) * distributed_batch_size * int(model.hidden_size))],
            (model.n_layers, distributed_batch_size, int(model.hidden_size))),
        requires_grad=False)
    model.memory_cn_senses = Parameter(
        torch.reshape(model.memory_hn_senses.flatten()[
                      0:((model.n_layers - 1) * distributed_batch_size * int(model.hidden_size))],
                      (model.n_layers - 1, distributed_batch_size, int(model.hidden_size))),
        requires_grad=False)
    model.hidden_state_bsize_adjusted = True

# Selecting the portion of memory tensor (determined by the layer size) that is used in the RNN iteration
def select_layer_memory(model, i, layer_rnn):
    if model.model_type.upper() == "LSTM":
        return (model.memory_hn.index_select(dim=0, index=model.select_first_indices[i].to(torch.int64)).
            index_select(dim=2, index=model.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64)),
            model.memory_cn.index_select(dim=0, index=model.select_first_indices[i].to(torch.int64)).
            index_select(dim=2, index=model.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64)))
    else: # GRU
        return  model.memory_hn.index_select(dim=0, index=model.select_first_indices[i].to(torch.int64)).\
            index_select(dim=2,index=model.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64))

# After execution of a RNN layer, save the new hidden state in the proper storage.
# May be only 'hidden' (if GRU) or also 'cells' (if LSTM)
def update_layer_memory(model, i, layer_rnn, hidden_i, cells_i=None):
    hidden_i_forcopy = hidden_i.index_select(dim=2,
                                             index=model.select_first_indices[0:layer_rnn.hidden_size].to(
                                                 torch.int64))
    hidden_i_forcopy = tfunc.pad(hidden_i_forcopy,
                                 pad=[0, (model.memory_hn.shape[2] - layer_rnn.hidden_size)]).squeeze()
    model.memory_hn[i].data.copy_(hidden_i_forcopy.clone())  # store h in memory
    if cells_i is not None:
        cells_i_forcopy = cells_i.index_select(dim=2,
                                               index=model.select_first_indices[0:layer_rnn.hidden_size].to(
                                                   torch.int64))
        cells_i_forcopy = tfunc.pad(cells_i_forcopy,
                                    pad=[0, model.memory_hn.shape[2] - layer_rnn.hidden_size]).squeeze()
        model.memory_cn[i].data.copy_(cells_i_forcopy.clone())
        return (hidden_i, cells_i)
    else:
        return hidden_i