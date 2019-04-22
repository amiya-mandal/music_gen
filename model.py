import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb


class RNN(nn.Module):
    def __init__(self, input_size=5000, hidden_size=1000, output_size=5000, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.liner = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.liner2 = nn.Linear(output_size, 1)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input_val, hidden):
        input_val = input_val.view(-1, 5000)
        input_val = self.liner(input_val)
        input_val = self.dropout(input_val)
        input_val = input_val.view(1, 1, -1)
        output, hidden = self.gru(input_val, hidden)
        output = self.decoder(output.view(1, -1))
        output = self.dropout(output)
        output = self.liner2(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))