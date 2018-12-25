import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import *

#class ConvLSTM(nn.Module):
# def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
#              batch_first=False, bias=True, return_all_layers=False):

class FullConvLstm(nn.Module):

    def __init__(self, num_layers, input_size = (224, 224), input_dim=3, hidden_dim=[64, 128],
                kernel_size = [(3, 3), (5, 5)], bias=True, batch_first=True):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.bias = bias

        self.convlstm_encode = ConvLSTM((self.height, self.width), self.input_dim, self.hidden_dim,
                                        self.kernel_size, self.num_layers, return_all_layers=True,
                                        bias=self.bias, batch_first=self.batch_first)
        self.convlstm_pred   = ConvLSTM((self.height, self.width), self.input_dim, self.hidden_dim,
                                        self.kernel_size, self.num_layers, return_all_layers=False,
                                        bias=self.bias, batch_first=self.batch_first)

        self.final_layer = nn.Conv2d(self.hidden_dim[-1], self.input_dim, [1, 1])

    def forward(self, input):
        encode_output, encode_state = self.convlstm_encode(input)
        encode_output               = F.relu(encode_output[-1])
        pred_output,   pred_state   = self.convlstm_pred(encode_output, encode_state)
        pred_output                 = F.relu(pred_output)
        final_output                = self.final_layer(pred_output)
        final_output                = F.relu(final_output)
        return final_output

criterion = nn.L1Loss()
