"""A simplified ACT (Action Chunking Transformer) policy, inspired by Zhao et al., "Learning
Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ALOHA/ACT, 2023).

Given one or more camera views plus proprioception+goal, predicts a chunk of future joint
targets in one shot. This is a deterministic simplification of the original paper (no CVAE
style/latent variable): since our training demonstrations come from a single deterministic
scripted expert rather than varied human teleoperation, there is little multimodality for a
latent variable to capture, so a direct regression transformer is used instead.

Each camera view is encoded by the *same* (shared-weight) CNN into one token; this lets the
model be used with either a single static camera, or a static + wrist camera pair, simply by
changing how many image tensors are passed to `forward()`.
"""

import torch
import torch.nn as nn

from src.demo_recorder import ACTION_DIM, STATE_DIM


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 84 -> 42
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 42 -> 21
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 21 -> 11
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 11 -> 6
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, img):
        # img: (B, H, W, 3) uint8 -> (B, 3, H, W) float in [0, 1]
        x = img.permute(0, 3, 1, 2).float() / 255.0
        x = self.net(x).flatten(1)
        return self.proj(x)


class ACTPolicy(nn.Module):
    def __init__(self, chunk_size=50, hidden_dim=256, n_heads=4, n_enc_layers=2, n_dec_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.state_proj = nn.Linear(STATE_DIM, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        self.query_pos = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)
        self.action_head = nn.Linear(hidden_dim, ACTION_DIM)

    def forward(self, images, state):
        """images: list of (B, 84, 84, 3) tensors (one per camera view), state: (B, STATE_DIM)
        -> (B, chunk_size, ACTION_DIM)"""
        batch = state.shape[0]
        img_toks = [self.image_encoder(img).unsqueeze(1) for img in images]  # each (B, 1, H)
        state_tok = self.state_proj(state).unsqueeze(1)  # (B, 1, H)
        memory = self.encoder(torch.cat(img_toks + [state_tok], dim=1))  # (B, n_views+1, H)

        queries = self.query_pos.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)  # (B, chunk_size, H)
        return self.action_head(decoded)
