import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim=256, patch_size=4, emb_dim=128, depth=3, heads=4):
        super().__init__()

        c, h, w = input_shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image must be divisible by patch size"

        self.patch_size = patch_size
        num_patches = (h // patch_size) * (w // patch_size)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            c, emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Projection to latent space
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, latent_dim),
            nn.ReLU()
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        x = x.float() / 255.0

        # Patchify
        x = self.patch_embed(x)  # [B, emb_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Global pooling (mean over patches)
        x = x.mean(dim=1)

        return self.fc(x)