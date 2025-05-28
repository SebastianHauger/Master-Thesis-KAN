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
            padding = ['custom'] + 8*[1] + ['custom']
        elif padding == 'asym_all':
            padding = ['custom'] * 10
        elif padding == 'uniform':
            padding = [1] * 10
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
        
       
        
        self.upsample3 = nn.ConvTranspose2d(embed_dims[0], embed_dims[0], 2, 2, padding=0)
        self.upsample2 = nn.ConvTranspose2d(embed_dims[0]//4, embed_dims[0]//4, 2, 2, padding=0)
        self.upsample1 = nn.ConvTranspose2d(embed_dims[0]//8, embed_dims[0]//8, 2, 2, padding=0)
        
        # change num patches to be correct... also think about changing other free params...
        self.patch_embed3 = ConvPatchEmbed(embed_dims[0], embed_dims[1], patch_s=2)
        self.patch_embed4 = ConvPatchEmbed(embed_dims[1], embed_dims[2], patch_s=2)
        self.inv_patch_embed4 = InvPatchEmbed(embed_dims[2], embed_dims[1], patch_s=2)
        self.inv_patch_embed3 = InvPatchEmbed(2*embed_dims[1], embed_dims[0], patch_s=2)
        
        # match decoder names with input names to make more sense.
        self.decoder3 = D_ConvLayer(embed_dims[0]*2, embed_dims[0]//4, padding[7])  
        self.decoder2 = D_ConvLayer(embed_dims[0]//2, embed_dims[0]//8, padding[8])  
        self.decoder1 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8, padding[9])
        
        self.final = nn.Conv2d(embed_dims[0]//8, 3, 1, 1) # last convolutional layer where we will process the output.

    def forward(self, x): 
        B = x.shape[0]
        ### Encoder
        
        ### Stage 1
        out = self.encoder1(x)
        t1 = out
        out = F.max_pool2d(out, 2, 2)
        
        ### Stage 2
        out = self.encoder2(out)
        t2 = out
        out = F.max_pool2d(out, 2, 2)
    
        ### Stage 3
        out = self.encoder3(out)
        t3 = out
        out = F.max_pool2d(out, 2, 2)  

        ### Stage 4 - KAN encoder 
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck
        out, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        ### Second bottleneck block. No reshapding required.
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # returns to original dimension
        out = self.inv_patch_embed4(out)
        
        ### Stage 4-- Kan decoder 
        out = torch.cat((out, t4), 1)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #Stage 3 -- decoder 
        out=self.inv_patch_embed3(out)
        out=self.upsample3(out)
        out = torch.cat((out, t3), 1)
        out = self.decoder3(out)
        
        # Stage 2 -- decoder 
        out = self.upsample2(out)
        out = torch.cat((out,t2), 1)
        out = self.decoder2(out)
        
        # Stage 1 -- decoder 
        out = self.upsample1(out)
        out = torch.cat((out,t1), 1)
        out = self.decoder1(out)
        
        # Final operations
        out = self.final(out)
        out = torch.add(out, x)
        return out