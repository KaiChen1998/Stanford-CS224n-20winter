#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embed_size, word_embed_size, kernel_size = 5, padding = 1):
        """
        Init CNN module  
        
        @param char_embed_size (int): character embedding size
        @param max_word_length (int)
        @param word_embed_size (int): conv1d out_channels
        @param kernel_size (int): default is 5
        @param padding (int): default is 1
        """
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size 
        self.word_embed_size = word_embed_size 
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv1d(self.char_embed_size, self.word_embed_size, self.kernel_size, padding = self.padding, bias = True)
        self.maxpooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x_reshape: torch.Tensor) -> torch.Tensor:
        """
        CNN character encoder: conv1d + maxpooling
        
        @param x_reshape (torch.Tensor): shape of [batch_size, char_embedding_size, num_words]
        @return x_conv_out (torch.Tensor): shape of [batch_size, word_embed_size]
        """
        x_conv = self.conv(x_reshape)
        x_conv_out = self.maxpooling(F.relu(x_conv))
        return x_conv_out.squeeze()

        ### END YOUR CODE
