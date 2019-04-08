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
SENTENCE_MAX_LEN = 20
INPUT_SIZE = 4096
VOCAB_SIZE = 10000
HIDDEN_SIZE = VOCAB_SIZE
LR = 0.01


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_seqs, pad_token):
    mid, (hidden1, cell1) = self.lstm1(input_seqs)
    new_mid = torch.cat((pad_token, mid), 1)
    outputs, (hidden2, cell2) = self.lstm2(new_mid, (hidden1, cell1))
    
    return outputs, (hidden1, cell1), (hidden2, cell2)
  
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = VOCAB_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_word, hidden1, cell1, hidden2, cell2):
    mid, _ = self.lstm1(input_seqs, (hidden1, cell1))
    new_mid = torch.cat((input_word, mid), 1)
    outputs, _ = self.lstm2(new_mid, (hidden2, cell2))
    
    return outputs
  
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    
  def forward(self, src, target, bos_token):
    hidden, cell = self.encoder(src)
    for t in range(1, SENTENCE_MAX_LEN):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      input = target[t]
      
    return outputs
