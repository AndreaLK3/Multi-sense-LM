import torch
from torch.nn import Parameter
from functools import wraps

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        # ***** My modification: we need to flatten_parameters(), so if in modern PyTorch we don't have bugs I would
        #       comment this out and keep the module's original flatten_parameters() function.
        # if issubclass(type(self.module), torch.nn.RNNBase):
        #     self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.training)
                # w = mask.expand_as(raw_w) * raw_w
                w = torch.nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                # w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
                w = torch.nn.Parameter(torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training))
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)


# # ***** My modification: since the WeightDrop of the author of the port
# # causes a UserWarning: RNN module weights not contiguous, etc. and clogs the GPU memory, I use my own.
#
# # During training, randomly zeroes some of the elements of the input tensor with probability p (Bernoulli distribution).
# # Functions are only pickle-able if they are defined at the top-level of a module,
# # so we create a class that is first initialized and then called as the forward()
# class ForwardWithDrop(object):
#     def __init__(self,weights_names_ls, module, dropout_p, original_module_forward):
#         self.weights_names_ls = weights_names_ls
#         self.module = module
#         self.dropout_p = dropout_p
#         self.original_module_forward = original_module_forward
#
#     def __call__(self, *args, **kwargs): # the function formerly known as "forward_new"
#         for name_param in self.weights_names_ls:
#             param = self.module._parameters.get(name_param)
#             param_with_droput = Parameter(torch.nn.functional.dropout(param, p=self.dropout_p, training=self.module.training),
#                                           requires_grad=param.requires_grad)
#             self.module._parameters.__setitem__(name_param, param_with_droput)
#
#         return self.original_module_forward(*args, **kwargs)


if __name__ == '__main__':
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')
