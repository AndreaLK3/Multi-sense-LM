import torch
import torch.nn as nn
import math


# ******* From the PyTorch implementation of Mogrifier-LSTM, at : https://github.com/fawazsammani/mogrifier-lstm-pytorch

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i + 1) % 2 == 0:
                h = (2 * torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2 * torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct

# *********************

# ********* Hyperparameters from the paper: "Mogrifier LSTM by G.Melis et al., 2020"

# input_size = 512
# hidden_size = 512
# vocab_size = 30
# batch_size = 4
# lr = 3e-3
# mogrify_steps = 5  # 5 steps give optimal performance according to the paper
# dropout = 0.5  # for simplicity: input dropout and output_dropout are 0.5. See appendix B in the paper for exact values
# tie_weights = True  # in the paper, embedding weights and output weights are tied
# betas = (0, 0.999)  # in the paper the momentum term in Adam is ignored
# weight_decay = 2.5e-4  # weight decay is around this value, see appendix B in the paper
# clip_norm = 10  # paper uses cip_norm of 10
#
# model = Model(input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-08, weight_decay=weight_decay)
#
# # seq of shape (batch_size, max_words)
# seq = torch.LongTensor([[8, 29, 18, 1, 17, 3, 26, 6, 26, 5],
#                         [8, 28, 15, 12, 13, 2, 26, 16, 20, 0],
#                         [15, 4, 27, 14, 29, 28, 14, 1, 0, 0],
#                         [20, 22, 29, 22, 23, 29, 0, 0, 0, 0]])
#
# outputs, hidden_states = model(seq)
# print(outputs.shape)
# print(hidden_states.shape)

# **** Modified from GitHub: "Here we provide an example of a model with two-layer Mogrifier LSTM."

class MOG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mogrify_steps, vocab_size, tie_weights, dropout):
        super(MOG_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(input_size, hidden_size, mogrify_steps)
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(hidden_size, hidden_size, mogrify_steps)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)
        if tie_weights:
            self.fc.weight = self.embedding.weight

    def forward(self, seq, max_len=10):

        embed = self.embedding(seq)
        batch_size = seq.shape[0]
        h1, c1 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
        h2, c2 = [torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)]
        hidden_states = []
        outputs = []
        for step in range(max_len):
            x = self.drop(embed[:, step])
            h1, c1 = self.mogrifier_lstm_layer1(x, (h1, c1))
            h2, c2 = self.mogrifier_lstm_layer2(h1, (h2, c2))
            out = self.fc(self.drop(h2))
            hidden_states.append(h2.unsqueeze(1))
            outputs.append(out.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)  # (batch_size, max_len, hidden_size)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_len, vocab_size)

        return outputs, hidden_states