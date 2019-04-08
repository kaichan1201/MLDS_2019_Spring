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
VOCAB_SIZE = 10000
HIDDEN_SIZE = VOCAB_SIZE
LR = 0.01


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_seqs):
    mid, (hidden1, cell1) = self.lstm1(input_seqs)
    outputs, (hidden2, cell2) = self.lstm2(mid, (hidden1, cell1))
    
    return outputs, (hidden1, cell1), (hidden2, cell2)
  
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = VOCAB_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_seqs, hidden1, cell1, hidden2, cell2):
    mid, _ = self.lstm1(input_seqs, (hidden1, cell1))
    outputs, _ = self.lstm2(mid, (hidden2, cell2))
    
    return outputs
  
