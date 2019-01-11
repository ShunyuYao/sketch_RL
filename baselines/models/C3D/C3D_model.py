import torch.nn as nn
import torch
import functools

class C3DVidPredNet(nn.Module):
    """
    The C3D network as described in [1].
    network for mmif gray face edges prediction
    """

    def __init__(self, use_dropout = False, norm = 'batch', use_lsgan=True):
        super(C3DVidPredNet, self).__init__()

        use_bias = not norm == 'batch'
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv5c = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        # self.conv5c = nn.Conv3d(1024, 1024, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)

        self.deconv1 = nn.ConvTranspose3d(64*2, 1, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(128*2, 64, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)
        self.deconv3 = nn.ConvTranspose3d(256*2, 128, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)
        self.deconv4 = nn.ConvTranspose3d(512*2, 256, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)
        self.deconv5a = nn.ConvTranspose3d(512*2, 512, kernel_size=(2, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)
        self.deconv5b = nn.ConvTranspose3d(512*2, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)
        self.deconv5c = nn.ConvTranspose3d(518, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2), bias=use_bias)

        self.downnorm2 = nn.BatchNorm3d(128)
        self.downnorm3 = nn.BatchNorm3d(256)
        self.downnorm4 = nn.BatchNorm3d(512)
        self.downnorm5 = nn.BatchNorm3d(512)
        self.downnorm5_1 = nn.BatchNorm3d(512)

        self.upnorm2 = nn.BatchNorm3d(64)
        self.upnorm3 = nn.BatchNorm3d(128)
        self.upnorm4 = nn.BatchNorm3d(256)
        self.upnorm5 = nn.BatchNorm3d(512)
        self.upnorm5_1 = nn.BatchNorm3d(512)

        self.dropout = nn.Dropout(p=0.5)

        self.downrelu = nn.LeakyReLU(0.2, True)
        self.uprelu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.use_dropout = use_dropout
        self.Sigmoid = nn.Sigmoid()
        self.use_lsgan = use_lsgan
        # self.softmax = nn.Softmax()

    def forward(self, x, feature):

        # print('original size:', x.size())
        encode1 = self.downrelu(self.conv1(x))
        # print('1 encode size:', encode1.size())
        encode2 = self.downrelu(self.downnorm2(self.conv2(encode1)))
        # print('2 encode size:', encode2.size())
        encode3 = self.downrelu(self.downnorm3(self.conv3(encode2)))
        # print('3 encode size:', encode3.size())
        encode4 = self.downrelu(self.downnorm4(self.conv4(encode3)))
        # print('4 encode size:', encode4.size())
        encode5 = self.downrelu(self.downnorm5(self.conv5a(encode4)))
        # print('5 encode size:', encode5.size())
        encode5_1 = self.downrelu(self.downnorm5_1(self.conv5b(encode5)))
        encode = self.downrelu(self.conv5c(encode5_1))
        encode = torch.cat([encode, feature], 1)
        # print('final size:', encode.size())

        decode5_1 = self.uprelu(self.deconv5c(encode))
        decode5_1 = torch.cat([encode5_1, decode5_1], 1)
        if self.use_dropout:
            decode5_1 = self.dropout(decode5_1)
        # print('5 decode size:', decode5.size())
        decode5 = self.uprelu(self.upnorm5_1(self.deconv5b(decode5_1)))

        decode5_fin = torch.Tensor([]).to(decode5.device)
        for i in range(decode5.size()[2]):
            decode5_temp = torch.cat([encode5, decode5[:, :, i].unsqueeze(2)], 1)
            decode5_fin = torch.cat([decode5_fin, decode5_temp], dim=2)

        decode5 = decode5_fin

        # decode4 = torch.cat([encode4, decode4], 1)
        if self.use_dropout:
            decode5 = self.dropout(decode5)

        decode4 = self.uprelu(self.upnorm5(self.deconv5a(decode5)))

        decode4_fin = torch.Tensor([]).to(decode4.device)
        for i in range(decode4.size()[2]):
            decode4_temp = torch.cat([encode4, decode4[:, :, i].unsqueeze(2)], 1)
            decode4_fin = torch.cat([decode4_fin, decode4_temp], dim=2)

        decode4 = decode4_fin

        # decode4 = torch.cat([encode4, decode4], 1)
        if self.use_dropout:
            decode4 = self.dropout(decode4)
        # print('4 decode size:', decode4.size())
        decode3 = self.uprelu(self.upnorm4(self.deconv4(decode4)))
        decode3_fin = torch.Tensor([]).to(decode3.device)
        for i in range(decode3.size()[2]):
            decode3_temp = torch.cat([encode3, decode3[:, :, i].unsqueeze(2)], 1)
            decode3_fin = torch.cat([decode3_fin, decode3_temp], dim=2)

        decode3 = decode3_fin
        # decode3 = torch.cat([encode3, decode3], 1)
        if self.use_dropout:
            decode3 = self.dropout(decode3)

        # print('3 decode size:', decode3.size())
        decode2 = self.uprelu(self.upnorm3(self.deconv3(decode3)))
        decode2_fin = torch.Tensor([]).to(decode2.device)
        for i in range(decode2.size()[2]):
            decode2_temp = torch.cat([encode2, decode2[:, :, i].unsqueeze(2)], 1)
            decode2_fin = torch.cat([decode2_fin, decode2_temp], dim=2)

        decode2 = decode2_fin
        # decode2 = torch.cat([encode2, decode2], 1)
        if self.use_dropout:
            decode2 = self.dropout(decode2)

        # print('2 decode size:', decode2.size())
        decode1 = self.uprelu(self.upnorm2(self.deconv2(decode2)))
        decode1_fin = torch.Tensor([]).to(decode1.device)
        for i in range(decode1.size()[2]):
            decode1_temp = torch.cat([encode1, decode1[:, :, i].unsqueeze(2)], 1)
            decode1_fin = torch.cat([decode1_fin, decode1_temp], dim=2)

        decode1 = decode1_fin
        # decode1 = torch.cat([encode1, decode1], 1)

        if self.use_dropout:
            decode1 = self.dropout(decode1)
        # print('1 decode size:', decode1.size())
        if self.use_lsgan:
            decode = self.tanh(self.deconv1(decode1))
        else:
            decode = self.Sigmoid(self.deconv1(decode1))
        # print('ori decode size:', decode.size())

        return decode

class C3D_discNet(nn.Module):
    """
    The C3D network as described in [1].
    network for mmif gray face edges prediction
    """

    def __init__(self, use_dropout = False, norm = 'batch', use_lsgan=False):
        super(C3D_discNet, self).__init__()

        use_bias = not norm == 'batch'
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)
        # self.conv5c = nn.Conv3d(512, 512, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=2, bias=use_bias)

        self.downnorm2 = nn.BatchNorm3d(128)
        self.downnorm3 = nn.BatchNorm3d(256)
        self.downnorm4 = nn.BatchNorm3d(512)
        self.downnorm5 = nn.BatchNorm3d(512)
        self.downnorm5_1 = nn.BatchNorm3d(512)

        self.downrelu = nn.LeakyReLU(0.2, True)
        self.use_lsgan = use_lsgan

        self.tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):

        # print('original size:', x.size())
        encode1 = self.downrelu(self.conv1(x))
        # print('1 encode size:', encode1.size())
        encode2 = self.downrelu(self.downnorm2(self.conv2(encode1)))
        # print('2 encode size:', encode2.size())
        encode3 = self.downrelu(self.downnorm3(self.conv3(encode2)))
        # print('3 encode size:', encode3.size())
        encode4 = self.downrelu(self.downnorm4(self.conv4(encode3)))
        # print('4 encode size:', encode4.size())
        encode5 = self.downrelu(self.downnorm5(self.conv5a(encode4)))
        # print('5 encode size:', encode5.size())
        # encode5_1 = self.downrelu(self.downnorm5_1(self.conv5b(encode5)))
        # encode = self.downrelu(self.conv5c(encode5_1))
        # encode = torch.cat([encode5, feature], 1)

        if self.use_lsgan:
            encode = self.tanh(encode5)
        else:
            encode = self.Sigmoid(encode5)

        # print('final size:', encode.size())

        return encode
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
