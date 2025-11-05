import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .basic import *
#medium : 8~10M
__all__ = ['vit_tiny', 'vit_small', 'vit_medium', 'vit_large', 'vit_base']

class PatchEmbedding(nn.Module):
    """将图像分割成patches并转换为embedding"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class Attention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(Basic):
    """轻量级Vision Transformer for CIFAR"""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, 
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4., 
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., mode='', weight=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier head
        if mode == 'norm':
            self.linear = NormLinear(embed_dim, num_classes)
        elif mode == 'fix':
            self.linear = FNormLinear(embed_dim, num_classes, weight)
        else:
            self.linear = nn.Linear(embed_dim, num_classes, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def get_body(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches+1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        
        # Norm
        x = self.norm(x)
        
        # Extract class token
        x = x[:, 0]  # [B, embed_dim]
        
        return x
    
    def forward(self, x):
        x = self.get_body(x)
        x = self.linear(x)
        return x


def vit_tiny(num_classes=10, mode='', weight=None):
    """ViT-Tiny: 192 dim, 6 layers, 3 heads"""
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=192, depth=6, num_heads=3, mlp_ratio=4., 
        qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.0, mode=mode, weight=weight
    )


def vit_small(num_classes=10, mode='', weight=None):
    """ViT-Small: 384 dim, 6 layers, 6 heads (~2M参数)"""
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=384, depth=6, num_heads=6, mlp_ratio=4., 
        qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.0, mode=mode, weight=weight
    )


def vit_medium(num_classes=10, mode='', weight=None):
    """
    ViT-Medium: 448 dim, 8 layers, 8 heads (~8-10M参数)
    介于Small和Large之间，适合中等规模实验
    """
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=448, depth=8, num_heads=8, mlp_ratio=4., 
        qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.0, mode=mode, weight=weight
    )


def vit_base(num_classes=10, mode='', weight=None):
    """ViT-Base: 768 dim, 12 layers, 12 heads"""
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
        qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.0, mode=mode, weight=weight
    )


def vit_large(num_classes=10, mode='', weight=None):
    """
    ViT-Large: 512 dim, 10 layers, 8 heads
    参数量约22M，与ResNet-34相近，适合对比实验
    """
    return VisionTransformer(
        img_size=32, patch_size=4, in_channels=3, num_classes=num_classes,
        embed_dim=512, depth=10, num_heads=8, mlp_ratio=4., 
        qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.0, mode=mode, weight=weight
    )


if __name__ == '__main__':
    model = vit_tiny(num_classes=10)
    x = torch.randn(5, 3, 32, 32)
    print(model(x).shape)  # Should be [5, 10]

