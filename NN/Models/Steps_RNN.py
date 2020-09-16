import torch
from torch.nn import Parameter, functional as tfunc

#############################
###### RNN operations #######
#############################

# The loop over the layers of a RNN
def rnn_loop(batch_input_signals, model, globals_or_senses_rnn):
    if globals_or_senses_rnn:
        rnn_ls = model.main_rnn_ls
    else:
        rnn_ls = model.senses_rnn_ls
    rnn_out = None
    input = batch_input_signals
    for i in range(model.n_layers):
        layer_rnn = rnn_ls[i]
        layer_rnn.flatten_parameters()
        # GRU
        rnn_out, hidden_i = \
            layer_rnn(input, select_layer_memory(model, i, layer_rnn, globals_or_senses_rnn))
        update_layer_memory(model, i, layer_rnn, hidden_i, globals_or_senses_rnn)

        input = rnn_out

    return rnn_out


# Reshaping the hidden state memories when we know the batch size allocated on the current GPU
def reshape_memories(distributed_batch_size, model):

    new_num_hidden_state_elems = model.n_layers * distributed_batch_size * model.hidden_size
    model.memory_hn = Parameter(torch.reshape(model.memory_hn.flatten()[0:new_num_hidden_state_elems],
                                              (model.n_layers, distributed_batch_size, model.hidden_size)),
                                requires_grad=False)
    model.memory_hn_senses = Parameter(
        torch.reshape(
            model.memory_hn_senses.flatten()[0:((model.n_layers) * distributed_batch_size * int(model.hidden_size))],
            (model.n_layers, distributed_batch_size, int(model.hidden_size))),
        requires_grad=False)
    model.hidden_state_bsize_adjusted = True

# Selecting the portion of memory tensor (determined by the layer size) that is used in the RNN iteration
def select_layer_memory(model, i, layer_rnn, globals_or_senses_rnn):
    memory = model.memory_hn if globals_or_senses_rnn else model.memory_hn_senses
    # GRU
    return  memory.index_select(dim=0, index=model.select_first_indices[i].to(torch.int64)).\
            index_select(dim=2,index=model.select_first_indices[0:layer_rnn.hidden_size].to(torch.int64))

# After execution of a RNN layer, save the new hidden state in the proper storage.
def update_layer_memory(model, i, layer_rnn, hidden_i, globals_or_senses_rnn):
    memory = model.memory_hn if globals_or_senses_rnn else model.memory_hn_senses
    hidden_i_forcopy = hidden_i.index_select(dim=2,
                                             index=model.select_first_indices[0:layer_rnn.hidden_size].to(
                                                 torch.int64))
    hidden_i_forcopy = tfunc.pad(hidden_i_forcopy,
                                 pad=[0, (memory.shape[2] - layer_rnn.hidden_size)]).squeeze()
    memory[i].data.copy_(hidden_i_forcopy.clone())  # store h in memory
    return hidden_i