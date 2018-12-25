import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from networks import *

writer = SummaryWriter()
input_nc = 3
output_nc = 3
ngf = 32
netG = 'unet_128'
model = define_G(input_nc, output_nc, ngf, netG, norm='batch',
         use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
model

class UnetConvLstmNet(nn.Module):
        def __init__(self, input_nc, output_nc, num_downs, ngf=64, input_size=(224, 224),
                     convlstm_layers = 2, convlstm_kernelSize = [3, 3],
                     norm_layer=nn.BatchNorm2d, use_dropout=False):
            super(UnetGenerator, self).__init__()
            self.height, self.width = input_size
            convlstm_input_H = self.height // (2 ** num_downs)
            convlstm_input_W = self.width // (2 ** num_downs)
            # construct unet structure

            lstm_block = nn.LSTM(ngf * 8, ngf * 8, dropout=use_dropout, num_layers=2, bidirectional=True)
            convlstm_block = ConvLSTM((convlstm_input_H, convlstm_input_W), ngf * 8, ngf * 8,
                                        convlstm_kernelSize, convlstm_layers, return_all_layers=True,
                                        bias=True, batch_first=True)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=convlstm_block, norm_layer=norm_layer, innermost=True)
            for i in range(num_downs - 5):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

            self.model = unet_block

        def forward(self, input):
            input.view(input.size(0), -1)
            return self.model(input)


nn.LSTM(10, 30, num_layers=2)
