import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.optim as optim
import torchvision.datasets as dsets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 100
INPUT_SIZE = 4096
HIDDEN_SIZE = VOCAB_SIZE
LR = 0.01


class Encoder(nn.Module):
  def __init__(self, vocab_size, embedding_size, output_size):
    super(Encoder, self).__init__()
    
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.LSTM_1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_layer=2)
    self.LSTM_2 = nn.LSTM(input_size = HIDDEN_SIZE, hidden_size = HIDDEN_SIZE)
   
  def forward(self, input_seqs, input_lengths, hidden=None):
    embedded = self.embedding(input_seqs)
    packed = pack_padded_sequence(embedded, input_lengths)
    packed_outputs, hidden = self.LSTM_1(packed, )
    packed_outputs, hidden 

