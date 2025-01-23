import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init 


"""Also copied from the paper on U-KAN. this file contains the different block structures that we will 
use inside our U-KAN. including the residual network structure. Note that I do not add patch embedding 
because i am not sure if it is relevant for this task"""

class DWConv(nn.Module):  # depth wise convolution ¨ 
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim) 

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ConvPatchEmbed(nn.Module):
    def __init__(self,in_ch, out_ch, patch_s, num_patches, dropout=0.2):
        """Note that patch_s needs to divide the spatial dimensions
        copied from Medium: https://medium.com/@fernandopalominocobo/demystifying-visual-transformers-with-pytorch-understanding-patch-embeddings-part-1-3-ba380f2aa37f"""
        super(ConvPatchEmbed, self).__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,
                             out_channels=out_ch, 
                             kernel_size=patch_s, 
                             stride=patch_s),
            nn.Flatten(2))
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,out_ch)), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.randn(size=(1, num_patches+1, out_ch)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, x):
        print(x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patch_embed(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        print(self.pos_embed.shape, x.shape)
        x = self.pos_embed + x
        x = self.dropout(x)
        return x, x.shape[1], x.shape[2] 
        