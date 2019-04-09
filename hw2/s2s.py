import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.optim as optim
import torchvision.datasets as dsets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import random
import math
import os

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 100
SENTENCE_MAX_LEN = 20
INPUT_SIZE = 4096
VOCAB_SIZE = 10000
HIDDEN_SIZE = VOCAB_SIZE
LR = 0.01
CLIP = 15


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE, batch_first = True)
   
  def forward(self, input_seqs, pad_token):
    mid, (hidden1, cell1) = self.lstm1(input_seqs)
    new_mid = torch.cat((pad_token, mid), 1)
    outputs, (hidden2, cell2) = self.lstm2(new_mid, (hidden1, cell1))
    
    return outputs, (hidden1, cell1), (hidden2, cell2)
  
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE, batch_first = True)
   
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
    input = bos_token
    for t in range(1, SENTENCE_MAX_LEN):
      output, hidden, cell = self.decoder(input, hidden, cell)
      outputs[t] = output
      input = target[t]
      
    return outputs

enc = Encoder()
dec = Decoder()

model = Seq2Seq(encoder,decoder).to(device)

#count param
def count_parametes(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

optimizer = optim.Adam(model.parameters(),lr= LR)

#!
loss_function = nn.CrossEntropyLoss()

def train(model, iterator, loss_function, clip):
  model.train()
  epoch_loss = 0

  for i, batch in enumerate(iterator):

    src = batch.src
    trg = batch.trg

    optimizer.zero_grad()

    output = model(src,trg)

    output = output[:].view(-1, output.shape[-1])
    trg = trg[:].view(-1)

    loss = loss_function(output,trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss/len(iterator)


def evaluate(model, iterator, loss_function, bos_token):
  model.eval()
  epoch_loss = 0

  with torch.no_grad():

    for i, batch in enumerate(iterator):

      src = batch.src
      trg = batch.trg

      output = model(src,trg,bos_token):

      output = output[:].view(-1, output.shape[-1])
      trg = trg[:].view(-1)

      loss = loss_function(output,trg)

      epoch_loss += loss.item()

    return epoch_loss/len(iterator)















