import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(tuple([tensor.shape[0]] + self.size))
    
class NiNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad, type_='down'):
        super(NiNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel 
        self.stride = stride
        self.pad = pad
        
        if type_ == 'down':
            self.block = nn.Sequential(nn.Conv2d(self.in_ch, self.out_ch, self.kernel, self.stride, 
                                                 self.pad, bias=False),
                                       nn.BatchNorm2d(self.out_ch),
                                       nn.ReLU(True))
        else:
            self.block = nn.Sequential(nn.ConvTranspose2d(self.in_ch, self.out_ch, self.kernel, self.stride, 
                                                 self.pad, bias=False),
                                       nn.BatchNorm2d(self.out_ch),
                                       nn.ReLU(True))
        
    def forward(self, x):
        return self.block(x)
    

class Encoder(nn.Module):
    def __init__(self, z_dim, num_channels, last_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.last_dim = last_dim
        
        self.net = nn.Sequential(NiNBlock(self.num_channels, 128, 4, 2, 1, 'down'),
                                 NiNBlock(128, 256, 4, 2, 1, 'down'),
                                 NiNBlock(256, 512, 4, 2, 1, 'down'),
                                 NiNBlock(512, 1024, 4, 2, 1, 'down'),
                                 View([-1]),
                                 nn.Linear(1024 * last_dim * last_dim, self.z_dim),)
        
    def forward(self, x):
        return self.net(x)
    

class Decoder(nn.Module):
    def __init__(self, z_dim, num_channels, first_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.num_channels = num_channels
        self.first_dim = first_dim

        self.net = nn.Sequential(nn.Linear(self.z_dim, 1024 * self.first_dim * self.first_dim),
                                 View([1024, self.first_dim, self.first_dim]),     
                                 NiNBlock(1024, 512, 4, 2, 1, 'up'),
                                 NiNBlock(512, 256, 4, 2, 1, 'up'),
                                 nn.ConvTranspose2d(256, self.num_channels, 3, 1, 1),
                                )
    def forward(self, x):
        return self.net(x)