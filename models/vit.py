import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, num_classes, patch_size=4, dim=128, depth=6, heads=8):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        self.embedding = nn.Linear(patch_size * patch_size * 3, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim * 4), depth
        )
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, self.num_patches, -1)
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0]) 