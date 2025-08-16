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

class FadeIn(nn.Module):
    def forward(self, old, new, alpha: float):
        return (1 - alpha) * old + alpha * new

def deconv(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, True)
    )

class ProgG_Gray(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fade = FadeIn()

        # 1×1   4×4
        self.start = nn.Sequential(
            nn.ConvTranspose2d(params['nz'],  params['ngf']*16, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(params['ngf']*16, affine=True),
            nn.LeakyReLU(0.2, True)
        )

        self.base_blocks = nn.Sequential(
            deconv(params['ngf']*16, params['ngf']*8),   # 4→8
            deconv(params['ngf']*8, params['ngf']*4),   # 8→16
            deconv(params['ngf']*4, params['ngf']*2),   # 16→32
        )

        # Progressive 新層（32→64、64→128）
        self.blocks = nn.ModuleList([
            deconv(params['ngf']*2, params['ngf']),  # 32→64
            deconv(params['ngf'], params['ngf']//2), # 64→128
        ])

        # toIMG：各解析度輸出灰階圖
        self.toIMG = nn.ModuleList([
            nn.Conv2d(params['ngf']*2, params['nc'], 1),       # 32×32
            nn.Conv2d(params['ngf'], params['nc'], 1),      # 64×64
            nn.Conv2d(params['ngf']//2, params['nc'], 1),    
         ])

    def forward(self, z, stage: int, alpha: float):
        """
        stage: 0=32x32, 1=64x64, 2=128x128
        alpha: fade-in 係數
        """
        x = self.start(z.view(z.size(0), -1, 1, 1))
        x = self.base_blocks(x)       # 到 32x32
        img = self.toIMG[0](x)        # 初始輸出

        if stage >= 1:
            x_new = self.blocks[0](x) # 32→64
            old_img = F.interpolate(self.toIMG[0](x), scale_factor=2, mode='nearest')
            new_img = self.toIMG[1](x_new)
            img = self.fade(old_img, new_img, alpha)
            x = x_new

        if stage >= 2:
            x_new = self.blocks[1](x) # 64→128
            old_img = F.interpolate(self.toIMG[1](x), scale_factor=2, mode='nearest')
            new_img = self.toIMG[2](x_new)
            img = self.fade(old_img, new_img, alpha)
            x = x_new

        return torch.tanh(img)


#spectral_norm
# Define the Discriminator Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

class FadeIn(nn.Module):
    def forward(self, old, new, alpha: float):
        return (1 - alpha) * old + alpha * new

def conv_down(in_ch, out_ch, use_sn=True):
    m = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
    m = SN(m) if use_sn else m
    return nn.Sequential(m, nn.LeakyReLU(0.2, inplace=True))

class MinibatchStd(nn.Module):
    def __init__(self, group=4, eps=1e-8):
        super().__init__(); self.group=group; self.eps=eps
    def forward(self, x):
        B,C,H,W = x.shape
        g = min(self.group, B)
        y = x.view(g, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt((y**2).mean(dim=0) + self.eps)
        y = y.mean(dim=[1,2,3], keepdim=True)
        y = y.repeat(g, 1, H, W)
        return torch.cat([x, y], dim=1)

class ProgD_Gray(nn.Module):
    # stage: 0=32x32, 1=64x64, 2=128x128
    def __init__(self, base=64, in_ch=1, use_sn=True, use_mbstd=True):
        super().__init__()
        self.fade = FadeIn()
        self.use_mbstd = use_mbstd

        self.fromIMG = nn.ModuleList([
            SN(nn.Conv2d(in_ch, base*1, 1)) if use_sn else nn.Conv2d(in_ch, base*1, 1),  # 32
            SN(nn.Conv2d(in_ch, base*2, 1)) if use_sn else nn.Conv2d(in_ch, base*2, 1),  # 64
            SN(nn.Conv2d(in_ch, base*4, 1)) if use_sn else nn.Conv2d(in_ch, base*4, 1),  # 128
        ])

        self.blocks = nn.ModuleList([
            conv_down(base*4, base*2, use_sn),  # 128 -> 64
            conv_down(base*2, base*1, use_sn),  # 64  -> 32
        ])

        tail = []
        if use_mbstd:
            tail.append(MinibatchStd())
            in_tail = base*1 + 1
        else:
            in_tail = base*1

        tail += [
            SN(nn.Conv2d(in_tail, base*1, 3, 1, 1, bias=False)) if use_sn else nn.Conv2d(in_tail, base*1, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(base*1, 1, 32, 1, 0, bias=False)) if use_sn else nn.Conv2d(base*1, 1, 32, 1, 0, bias=False)
        ]
        self.final = nn.Sequential(*tail)

    def forward(self, x, stage: int, alpha: float):
        if stage == 0:  # 32
            feat32 = self.fromIMG[0](x)
            return self.final(feat32).view(-1)

        if stage == 1:  # 64 -> mix 到 32
            new32 = self.blocks[1]( self.fromIMG[1](x) )  # 64->32
            old32 = self.fromIMG[0]( F.avg_pool2d(x, 2) ) # 64->32 (影像下採樣)
            feat32 = self.fade(old32, new32, alpha)
            return self.final(feat32).view(-1)

        if stage == 2:  # 128 -> mix 到 32
            h64   = self.blocks[0]( self.fromIMG[2](x) )  # 128->64
            new32 = self.blocks[1]( h64 )                 # 64->32

            old64 = self.fromIMG[1]( F.avg_pool2d(x, 2) ) # 128->64
            old32 = self.blocks[1]( old64 )               # 64->32

            feat32 = self.fade(old32, new32, alpha)
            return self.final(feat32).view(-1)

