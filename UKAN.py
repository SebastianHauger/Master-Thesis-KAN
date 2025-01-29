import torch
from torch import nn 
from convLayers import *
from KANlayers import KANBlock


class UKAN(nn.Module):
    def __init__(self, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])
        
        # change num patches to be correct... also think about changing other free params...
        self.patch_embed3 = ConvPatchEmbed(embed_dims[0], embed_dims[1], patch_s=2, num_patches=210, dropout=0.1)
        self.patch_embed4 = ConvPatchEmbed(embed_dims[1], embed_dims[2], patch_s=2, num_patches=210, dropout=0.1)
        
        
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)
        self.final = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x): 
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        # print(f"shape {x.shape} size = {x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]}")
        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")   

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]}")
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")

        ### Bottleneck

        out, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))
        # print(f"shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        # print(f"shape {out.shape}")
        ### Stage 3
        out = self.dnorm3(out)
        # print(f"shape {out.shape}")
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"shape {out.shape}")
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"shape {out.shape}")
        out = torch.add(out,t3)
        # print(f"shape {out.shape}")
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        # print(f"shape {out.shape}")
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # print(f"shape {out.shape}")
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"shape {out.shape}")
        out = torch.add(out,t2)
        # print(f"shape {out.shape}")
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"shape {out.shape}")
        out = torch.add(out,t1)
        # print(f"shape {out.shape}")
        out = F.sigmoid(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"shape {out.shape}")
        out = self.final(out)
        
        out = torch.add(out, x)
        # print(f"shape {out.shape}") 
        return out