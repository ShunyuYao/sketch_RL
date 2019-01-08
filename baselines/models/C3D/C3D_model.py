import torch.nn as nn
import torch
import functools

class C3DVidPredNet(nn.Module):
    """
    The C3D network as described in [1].
    network for mmif gray face edges prediction
    """

    def __init__(self, use_dropout = False):
        super(C3DVidPredNet, self).__init__()

        self.conv1 = nn.Conv3d(2, 64, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        # self.conv5c = nn.Conv3d(1024, 1024, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)

        self.deconv1 = nn.ConvTranspose3d(64, 1, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv5a = nn.ConvTranspose3d(512, 512, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv5b = nn.ConvTranspose3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))

        self.dropout = nn.Dropout(p=0.5)

        self.downrelu = nn.LeakyReLU(0.2, True)
        self.uprelu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.use_dropout = use_dropout
        # self.softmax = nn.Softmax()

    def forward(self, x):

        # print('original size:', x.size())
        encode = self.downrelu(self.conv1(x))
        # print('1 encode size:', encode.size())
        encode = self.downrelu(self.conv2(encode))
        # print('2 encode size:', encode.size())
        encode = self.downrelu(self.conv3(encode))
        # print('3 encode size:', encode.size())
        encode = self.downrelu(self.conv4(encode))
        # print('4 encode size:', encode.size())
        encode = self.downrelu(self.conv5a(encode))
        # print('5 encode size:', encode.size())
        encode = self.downrelu(self.conv5b(encode))
        # print('final size:', encode.size())

        decode = self.uprelu(self.deconv5b(encode))
        if self.use_dropout:
            decode = self.dropout(decode)
        # print('5 decode size:', decode.size())
        decode = self.uprelu(self.deconv5a(decode))
        if self.use_dropout:
            decode = self.dropout(decode)
        # print('4 decode size:', decode.size())
        decode = self.uprelu(self.deconv4(decode))
        if self.use_dropout:
            decode = self.dropout(decode)
        # print('3 decode size:', decode.size())
        decode = self.uprelu(self.deconv3(decode))
        if self.use_dropout:
            decode = self.dropout(decode)
        # print('2 decode size:', decode.size())
        decode = self.uprelu(self.deconv2(decode))
        if self.use_dropout:
            decode = self.dropout(decode)
        # print('1 decode size:', decode.size())
        decode = self.tanh(self.deconv1(decode))
        # print('ori decode size:', decode.size())

        return decode

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
Proceedings of the IEEE international conference on computer vision. 2015.
"""
