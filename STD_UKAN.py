import torch
from torch import nn 
from convLayers import *
from KANlayers import KANBlock




class UKAN(nn.Module):
    def __init__(self, embed_dims=[256, 320, 512], drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1], padding='uniform', **kwargs):
        """Let padding be in ['uniform', 'asym_1', 'asym_all'] depending on if we 
        want to use the custom padding in none of the layers, in the first and last layer only, 
        or if we want to use it for all convolutional layers."""
        super().__init__()
        if padding == 'asym_1':
            padding = ['custom'] + 11*[1] + ['custom']
        elif padding == 'asym_all':
            padding = ['custom'] * 13
        elif padding == 'uniform':
            padding = [1] * 13
        else: raise AttributeError("padding is not in ['uniform', 'asym_1', 'asym_all']")
            
        kan_input_dim = embed_dims[0]
        self.encoder1 = ConvLayer(3, kan_input_dim//8, padding=padding[0])  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4, padding=padding[1])  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim, padding=padding[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[2])
        self.dnorm4 = norm_layer(2*embed_dims[1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, padding=padding[3]
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, padding=padding[4]
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2], dim2 =embed_dims[2],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, padding=padding[5]
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=2*embed_dims[1],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, padding=padding[6]
            )])
        
        # change num patches to be correct... also think about changing other free params...
        self.patch_embed3 = ConvPatchEmbed(embed_dims[0], embed_dims[1], patch_s=2)
        self.patch_embed4 = ConvPatchEmbed(embed_dims[1], embed_dims[2], patch_s=2)
        self.inv_patch_embed4 = InvPatchEmbed(embed_dims[2], embed_dims[1], patch_s=2)
        self.inv_patch_embed3 = InvPatchEmbed(2*embed_dims[1], embed_dims[0], patch_s=2)
        
        # match decoder names with input names to make more sense.
        self.decoder3 = D_ConvLayer(embed_dims[2], embed_dims[0]//4, padding[7])  
        self.decoder2 = D_ConvLayer(embed_dims[0]//2, embed_dims[0]//8, padding[8])  
        self.decoder1 = D_ConvLayer(embed_dims[0]//4, 3, padding[9])

    def forward(self, x): 
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        # print(f"Stage 1shape {x.shape} size = {x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3]}")
        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        # print(f"Stage 2 shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        # print(f"Stage 3 shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        # print(f"Stage 4 shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")   

        ### Stage 4 - KAN encoder 
        out, H, W = self.patch_embed3(out)
        # print(f"Stage 4' shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]}")
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        # print(f"Bottleneck shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]}")

        
        ### Bottleneck
        out, H, W= self.patch_embed4(out)
        # print(f"Bottleneck' shape {out.shape} size = {out.shape[0]*out.shape[1]*out.shape[2]}")
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        ### Second bottleneck block. No reshapding required.
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # returns to original dimension
        # here we need to reverse the bottom patch embedding in a meaningful way..
        out = self.inv_patch_embed4(out)
        ### Stage 4-- Kan decoder 
        # print(f"Stage 4 dec shape {out.shape}")
        out = torch.cat((out, t4), 1)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        # print(f"Stage 4' shape {out.shape}")
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        # print(f"after second decKAN shape {out.shape}")
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        #Stage 3 -- decoder 
        out=self.inv_patch_embed3(out)
        out = torch.cat((out, t3), 1)
        # print(f"Stage 3 shape {out.shape}")
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"Stage 3 out shape {out.shape}")
        
        # Stage 2 -- decoder 
        out = torch.cat((out,t2), 1)
        # print(f"Stage 2 dec shape {out.shape}")
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"Stage 2 out shape {out.shape}")
        
        # Stage 1 -- decoder 
        out = torch.cat((out,t1), 1)
        # print(f"Stage 1 dec shape {out.shape}")
        out = F.tanh(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        # print(f"Stage 1 out shape {out.shape}")
        
        # perhaps we should have a last convolutional layer since we have interpolated???
        
        # Final stage. Add input so output will model grad(f)
        out = torch.add(out, x)
        # # print(f"shape {out.shape}") 
        return out