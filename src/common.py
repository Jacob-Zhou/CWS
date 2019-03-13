# -*- coding: utf-8 -*-

import time

import torch

padding_str = '<-PAD->'
padding_idx = 0
unknown_str = '<-UNK->'
unknown_idx = 1

abs_max_score = 1e5
eps_ratio = 1e-10


def coarse_equal_to(self, a, b):
    eps = eps_ratio * abs(b)
    return b + eps >= a >= b - eps


def drop_input_word_tag_emb_independent(word_embeddings, tag_embeddings, dropout):
    assert (dropout >= 0.33 - 1e-5) and dropout <= (0.33 + 1e-5)
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = compose_drop_mask(
        word_embeddings, (batch_size, seq_length), dropout)
    tag_masks = compose_drop_mask(
        tag_embeddings, (batch_size, seq_length), dropout)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale  # DO NOT understand this part.
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    return word_embeddings, tag_embeddings


def compose_drop_mask(x, size, dropout):
    # old way (before torch-0.4)
    # in_drop_mask = x.data.new(batch_size, input_size).fill_(1 - self.dropout_in) # same device as x
    # in_drop_mask = Variable(torch.bernoulli(in_drop_mask), requires_grad=False)
    drop_mask = x.new_full(size, 1 - dropout, requires_grad=False)
    return torch.bernoulli(drop_mask)
    # no need to expand in_drop_mask
    # in_drop_mask = torch.unsqueeze(in_drop_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
    # x = x * in_drop_mask


def drop_sequence_shared_mask(inputs, dropout):
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = compose_drop_mask(
        inputs, (batch_size, hidden_size), dropout) / (1 - dropout)
    # drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    return inputs * drop_masks  # should be broadcast-able


def get_time_str():
    return time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time()))
