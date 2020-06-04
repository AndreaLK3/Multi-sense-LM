import torch
import gc
import numpy as np

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


# Modified: I add GPU memory profiling to try to find the cause for a CUDA-OutOfMemory
# prints currently alive Tensors and Variables
def gpu_memory_profiling():
    torch.cuda.empty_cache()
    # CURRENT_DEVICE = 'cpu' if not (torch.cuda.is_available()) else 'cuda:' + str(torch.cuda.current_device())
    # CURRENT_DEVICE is currently unneeded because awd-lstm uses 1 GPU
    for obj in gc.get_objects():
        try:
             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print((type(obj), obj.size()))
        except:
            pass

# Added
def print_model_named_params(model):
    parameters_list = [(name, param.shape, param.dtype, param.requires_grad) for (name, param) in model.named_parameters()]
    print("Named parameters of the model")
    print('\n'.join([str(p) for p in parameters_list]))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters=" + str(params))