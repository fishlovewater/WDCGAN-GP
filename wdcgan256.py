import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    classname = w.__class__.__name__
    if classname.lower().find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.lower().find('batchnorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)
    elif classname.lower().find('layernorm') != -1:
        nn.init.constant_(w.weight.data, 1.0)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()#繼承nn.Module
        #(inputsize-1)*stride-padding*2+kernalsize
        #nn.ConvTranspose2d(in, out , kernalsize, stride, padding, bias=False)
        self.label_emb = nn.Embedding(params['class'], params['nz'])
        self.net = nn.Sequential(
            nn.ConvTranspose2d(params['nz']*2, params['ngf'] * 16, 4, 1, 0, bias=False),  # 1x1 -> 4x4 1024
            nn.BatchNorm2d(params['ngf'] * 16),#let mean=0 sigma=1
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 16, params['ngf'] * 8, 4, 2, 1, bias=False),  # 4 -> 8 512
            nn.BatchNorm2d(params['ngf'] * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 8, params['ngf'] * 4, 4, 2, 1, bias=False),  # 8 -> 16 256
            nn.BatchNorm2d(params['ngf'] * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 4, params['ngf'] * 2, 4, 2, 1, bias=False),  # 16 -> 32 128
            nn.BatchNorm2d(params['ngf'] * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'] * 2, params['ngf'], 4, 2, 1, bias=False),  # 32 -> 64 64
            nn.BatchNorm2d(params['ngf']),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf'], params['ngf']//2, 4, 2, 1, bias=False),  # 64 -> 128 32
            nn.BatchNorm2d(params['ngf']//2),
            nn.ReLU(True),

            nn.ConvTranspose2d(params['ngf']//2, params['nc']-1, 4, 2, 1, bias=False),  # 128 -> 256 16
            nn.Tanh()
        )

    def forward(self, x, label):
        batch_size = x.size(0)
        label_map_with_pic = self.label_emb(label).view(batch_size, 100, 1, 1)
        x = torch.cat([x, label_map_with_pic], dim=1) 
        x = x.view(batch_size, -1, 1, 1)
        return self.net(x)


# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        #We found that using batchnorm on the generator and discriminator helped
        # , with the exception of the output layer of the generator and the input layer of the discriminator.
        #(inputsize+2*padding-kernalsize)//stride+1
        self.imsize = params['imsize']
        #轉換編碼 模擬label->512*512 取代onehot
        self.label_emb = nn.Embedding(params['class'], params['imsize'] * params['imsize'])
        self.dnet= nn.Sequential(
            nn.Conv2d(params['nc'], params['ndf'], 4, 2, 1, bias=False),  #256 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(params['ndf'], params['ndf']*2, 4, 2, 1, bias=False),  #128 64
            nn.LayerNorm([params['ndf']*2, 64, 64]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf']*2, params['ndf']*4, 4, 2, 1, bias=False), #64 32
            nn.LayerNorm([params['ndf']*4, 32, 32]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf']*4, params['ndf']*8, 4, 2, 1, bias=False),  #32 16
            nn.LayerNorm([params['ndf']*8, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(params['ndf']*8, params['ndf']*8, 4, 2, 1, bias=False),  # 16 8
            nn.LayerNorm([params['ndf']*8, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(params['ndf']*8, params['ndf']*8, 4, 2, 1, bias=False),  
            # nn.LayerNorm([params['ndf']*8, 4, 4]),
            # nn.LeakyReLU(0.2, inplace=True),

        )
        self.output = nn.Sequential(
             nn.Conv2d(params['ndf']*8, 1, 8, 1, 0, bias=False)  # 8×8 → 1×1
        )

    def forward(self, x, label):
        label_batch_size = label.size(0)
        label_map_with_pic = self.label_emb(label).view(label_batch_size, 1, self.imsize, self.imsize)
        x = torch.cat([x, label_map_with_pic], dim=1) 
        x = self.dnet(x)
        x = self.output(x)
        return x.view(-1)
    def feature_extractor(self, x, label):
        label_batch_size = label.size(0)
        label_map_with_pic = self.label_emb(label).view(label_batch_size, 1, self.imsize, self.imsize)
        x = torch.cat([x, label_map_with_pic], dim=1) 
        x = self.dnet(x)
        return x.view(x.size(0), -1)

