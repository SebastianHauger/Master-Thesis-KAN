import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init 


"""Also copied from the paper on U-KAN. this file contains the different block structures that we will 
use inside our U-KAN. including the residual network structure. """


class CustomPad2d(nn.Module):
    def __init__(self, pad_x=2, pad_y=3):
        super(CustomPad2d, self).__init__()
        self.pad_x = pad_x
        self.pad_y = pad_y 
        
    def forward(self, x):
        x = F.pad(x, (self.pad_x, self.pad_x, 0, 0), mode='circular')  # Circular padding for x direction
        x = F.pad(x, (0, 0, self.pad_y, self.pad_y), mode='reflect')  # Reflection padding for y direction
        return x
        

class DWConv(nn.Module):  # depth wise convolution ¨ 
    def __init__(self, dim=768, padding=1, pad_x=1, pad_y=1):
        super(DWConv, self).__init__()
        if padding=='custom':
            self.dwconv = nn.Sequential(CustomPad2d(pad_x, pad_y), nn.Conv2d(dim, dim, 3, 1, padding='valid', bias=True, groups=dim))
        else:   
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, padding=padding, bias=True, groups=dim) 
            

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class DW_bn_relu(nn.Module):
    def __init__(self, dim=768, padding=1, pad_x=1, pad_y=1):
        super(DW_bn_relu, self).__init__()
        if padding=='custom':
            self.dwconv = nn.Sequential(CustomPad2d(pad_x, pad_y), nn.Conv2d(dim, dim, 3, 1, padding='valid', bias=True, groups=dim))
        else:   
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, padding=padding, bias=True, groups=dim) 
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
    def __init__(self, in_ch, out_ch, padding=1, pad_x=1, pad_y=1):
        super(ConvLayer, self).__init__()
        if padding=='custom':
            self.conv = nn.Sequential(
                    CustomPad2d(pad_x, pad_y),
                    nn.Conv2d(in_ch, out_ch, 3, padding='valid'),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    CustomPad2d(pad_x, pad_y),
                    nn.Conv2d(out_ch, out_ch, 3, padding='valid'),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )  
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ) 

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1, pad_x=1, pad_y=1):
        super(D_ConvLayer, self).__init__()
        if padding=='custom':
            self.conv = nn.Sequential(
                    CustomPad2d(pad_x, pad_y),
                    nn.Conv2d(in_ch, in_ch, 3, padding='valid'),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    CustomPad2d(pad_x, pad_y),
                    nn.Conv2d(in_ch, out_ch, 3, padding='valid'),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )  
        else:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, padding=padding),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 3, padding=padding),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )  

    def forward(self, input):
        return self.conv(input)


class ConvPatchEmbed(nn.Module):
    def __init__(self,in_ch, out_ch, patch_s): 
        super(ConvPatchEmbed, self).__init__()
        self.project = nn.Conv2d(in_channels=in_ch,
                                     out_channels=out_ch, 
                                     kernel_size=patch_s, 
                                     stride=patch_s, padding='valid')
        self.patch_size = patch_s
        
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.project(x)
        B2, C2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2) # from timm, BCHW -> BNC
        print(H/H2, W/W2)
        H = int(H/self.patch_size)
        W = int(W/self.patch_size)
        # print(H, W)
        return x, H, W


class InvPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, patch_s):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=patch_s, stride=patch_s, padding=(0,0))
    
    def forward(self, x):
        return self.deconv(x)
    
    
