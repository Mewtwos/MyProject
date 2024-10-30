from timm.models.vision_transformer import Block
from torch import nn
import torch
from .pos_embed import get_2d_sincos_pos_embed


class MAE_Decoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=256, out_dim=27, num_patches=49, depth=1, num_heads=8, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.embed = nn.Linear(inp_dim, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed tokens   
        x = self.embed(x) #输入格式[b, patch_num, feature]

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, L, d]

        # predictor projection
        H = W = int(self.num_patches**0.5)
        x = x[:, 1:].transpose(1, 2).reshape(x.size(0), -1, H, W) # [b, 256, patch_size, patch_size]

        return x