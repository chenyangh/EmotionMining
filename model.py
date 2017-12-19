import torch
import torch.nn as nn
from data_helper import *


class CNN_LSTM(nn.Module):
    def __init__(self, n_words, emb_dim, target_dim,
         ):
        super(CNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(n_words, emb_dim)





