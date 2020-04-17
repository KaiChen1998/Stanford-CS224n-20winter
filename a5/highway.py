#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size):
        """
        Init the highway module
        
        @param word_embed_size (int): word embedding size
        """
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        Highway layer: output = gate * proj + (1 - gate) * conv
        
        @param x_conv (torch.Tensor): output from maxpooling, shape of [batch_size, word_embedding_size]
        @return x_highway (torch.Tensor): shape of [batch_size, word_embedding_size]
        """
        x_proj = F.relu(self.proj(x_conv))
        x_gate = torch.sigmoid(self.gate(x_conv))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv

        return x_highway

    ### END YOUR CODE

