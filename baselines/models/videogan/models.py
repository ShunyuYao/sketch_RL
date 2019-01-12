import torch
import torch.nn as nn
import os


class G_video(nn.Module):
    def __init__(self, input_channels):
        super(G_video, self).__init__()
        self.model = nn.Sequential()
