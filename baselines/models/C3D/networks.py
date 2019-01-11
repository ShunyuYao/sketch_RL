import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D_patch(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

class C3D_Discriminator(nn.Module):
    def __init__(self):
        super(C3D_Discriminator, self).__init__()
        self.model = nn.Sequential( # [-1, 3, 32, 64, 64]
                self._conv3d(3, 128), #[-1, 64, 16, 32, 32]
                self._lrelu(0.2),
                self._conv3d(128,256), #[-1, 126,8,16,16]
                self._batchNorm5d(256, 1e-3),
                self._lrelu(0.2),
                self._conv3d(256,512), #[-1,256,4,8,8]
                self._batchNorm5d(512, 1e-3),
                self._lrelu(0.2),
                self._conv3d(512,1024), #[-1,512,2,4,4]
                self._batchNorm5d(1024,1e-3),
                self._lrelu(0.2),
                self._conv3d(1024,2, (2,4,4), (1,1,1), (0,0,0)) #[-1,2,1,1,1] because (2,4,4) is the kernel size
                )
        #self.mymodules = nn.ModuleList([nn.Sequential(nn.Linear(2,1), nn.Sigmoid())])

def _conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)

def _conv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def _deconv2d_first(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (4,4))

def _deconv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def _deconv3d_first(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,4,4))

def _deconv3d_video(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,1,1))

def _deconv3d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True)


def _batchNorm4d(num_features, eps = 1e-5): #input: N, C, H, W
    return nn.BatchNorm2d(num_features, eps = eps)

def _batchNorm5d(num_features, eps = 1e-5): #input: N, C, D, H, W
    return nn.BatchNorm3d(num_features, eps = eps)

def _relu(inplace = True):
    return nn.ReLU(inplace)

def _lrelu(negative_slope = 0.2, inplace = True):
    return nn.LeakyReLU(negative_slope, inplace)

    def forward(self, x):
        out = self.model(x).squeeze()
        #out = self.mymodules[0](out)
        return out
