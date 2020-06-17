import numpy as np

import torch
from torch import nn


class EmbeddingDropout(nn.Module):
  """
  Embedding Layer.
  If embedding_dropout != 0 we apply dropout to word 'types' not 'tokens' as suggested
  in the paper https://arxiv.org/pdf/1512.05287.pdf.
  We first map the input sequences to the corresponding embeddings (from |V| -> embedding_dim)
  and THEN apply dropout.
  """

  def __init__(self, num_embeddings, embedding_dim, embedding_dropout=0.):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.dropoute = embedding_dropout

    self.embed = nn.Embedding(num_embeddings=self.num_embeddings,
                              embedding_dim=self.embedding_dim)

  def forward(self, words):
    if self.dropoute and self.training:
      mask = self.embed.weight.data.new().resize_((self.embed.weight.size(0), 1)).bernoulli_(
        1 - self.dropoute).expand_as(
        self.embed.weight) / (1 - self.dropoute)
      masked_embed_weight = mask * self.embed.weight
    else:
      masked_embed_weight = self.embed.weight

    padding_idx = self.embed.padding_idx  # be careful here to use the same 'padding_idx' name
    if padding_idx is None:
      padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, self.embed.max_norm, self.embed.norm_type,
                                      self.embed.scale_grad_by_freq, self.embed.sparse
                                      )
    return X

def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    masked_embed_weight = mask * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1

  X = torch.nn.functional.embedding(words, masked_embed_weight,
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return X

if __name__ == '__main__':
  V = 50
  h = 4
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(V, h)

  words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
  words = torch.LongTensor(words)

  origX = embed(words)
  X = embedded_dropout(embed, words)

  print(origX)
  print(X)
