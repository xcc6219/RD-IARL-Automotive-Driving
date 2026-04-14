# -*- coding: utf-8 -*-
"""
DDPG inference-only module.

Defines the neural network architectures (TinyVisionStem, ConvFeatureExtractor,
Actor) and a lightweight DDPGAgent class that loads pretrained weights and
performs action selection without any training infrastructure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def _group_norm(num_channels, num_groups=None):
    """Select a compatible GroupNorm configuration for the given channel count."""
    if num_groups is None:
        for g in [8, 4, 2, 1]:
            if num_channels % g == 0:
                num_groups = g
                break
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Ensure an image tensor is in (N, C, H, W) format."""
    if x.dim() == 3:
        if x.shape[0] in (1, 3, 4):
            pass
        elif x.shape[-1] in (1, 3, 4):
            x = x.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Unrecognized 3D image shape: {x.shape}")
        x = x.unsqueeze(0)
    elif x.dim() == 4:
        if x.shape[1] in (1, 3, 4):
            pass
        elif x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unrecognized 4D image shape: {x.shape}")
    else:
        raise ValueError(f"Unsupported image tensor shape: {x.shape}")
    return x


def _to_tensor(x, device):
    """Convert a numpy array to a float32 tensor on the specified device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True).contiguous()
    return torch.from_numpy(x).float().contiguous().to(device, non_blocking=True)


def xyz_to_device(state, device):
    """Unpack observation dict and move ego-state, BEV, and semantic tensors to device."""
    x = _to_tensor(state['state'], device).float()
    y = _to_tensor(state['birdseye'], device).float()
    z = _to_tensor(state['semantic'], device).float()

    if x.dim() == 1:
        x = x.unsqueeze(0)

    y = _ensure_nchw(y)
    z = _ensure_nchw(z)
    return x, y, z


class TinyVisionStem(nn.Module):
    """Lightweight 3-layer CNN for encoding a single image stream."""

    def __init__(self, in_ch, downsample_factor=1, c1=16, c2=32, c3=32):
        super().__init__()
        self.pre_down = None
        if int(downsample_factor) > 1:
            self.pre_down = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, stride=2, padding=1),
            _group_norm(c1),
            nn.LeakyReLU(),

            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            _group_norm(c2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            _group_norm(c3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x):
        if self.pre_down is not None:
            x = self.pre_down(x)
        return self.net(x)


class ConvFeatureExtractor(nn.Module):
    """Dual-stream vision encoder fusing BEV and semantic inputs with ego state."""

    def __init__(self, img_in, device, env_dim=32, dim_num=14, downsample_factor=1, use_layernorm=True):
        super().__init__()
        self.device = device
        self.env_dim = env_dim
        self.dim_num = dim_num

        self.conv_bev = TinyVisionStem(img_in, downsample_factor=downsample_factor)
        self.conv_seg = TinyVisionStem(img_in, downsample_factor=downsample_factor)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        env_layers = [nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, self.env_dim), nn.LeakyReLU()]
        if use_layernorm:
            env_layers.append(nn.LayerNorm(self.env_dim))
        self.env_proj = nn.Sequential(*env_layers)

    def forward(self, state):
        ego, y, z = xyz_to_device(state, self.device)

        bev_fm = self.conv_bev(y)
        seg_fm = self.conv_seg(z)
        bev_vec = self.gap(bev_fm).squeeze(-1).squeeze(-1)
        seg_vec = self.gap(seg_fm).squeeze(-1).squeeze(-1)

        env_latent = self.env_proj(torch.cat([bev_vec, seg_vec], dim=-1))
        fused = torch.cat([ego, env_latent], dim=-1)
        return fused


class Actor(nn.Module):
    """MLP policy network that outputs 2D continuous actions (throttle delta, steer delta)."""

    def __init__(self, input_features):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.actions = nn.Linear(8, 2)

    def forward(self, fused):
        x = F.leaky_relu(self.fc1(fused))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.tanh(self.actions(x))


class DDPGAgent:
    """
    Inference-only DDPG agent.

    Loads pretrained feature extractor and actor weights, then performs
    deterministic action selection using residual action blending.
    """

    def __init__(self, params):
        self.opt = params

        env_dim = 32
        dim_num = 14
        fused_dim = dim_num + env_dim
        downsample_factor = int(getattr(self.opt, 'downsample_factor', 1))

        self.feature_extractor = ConvFeatureExtractor(
            img_in=3, device=self.opt.device,
            env_dim=env_dim, dim_num=dim_num, downsample_factor=downsample_factor
        ).to(self.opt.device)

        self.actor_net = Actor(input_features=fused_dim).float().to(self.opt.device)

        self.alpha = float(getattr(self.opt, 'residual_alpha', 0.8))

    def start_episode(self):
        """Called at the beginning of each episode."""
        pass

    def select_action(self, state):
        """Compute a deterministic action from the current observation."""
        self.actor_net.eval()
        self.feature_extractor.eval()

        with torch.no_grad():
            feat = self.feature_extractor(state)
            if not torch.isfinite(feat).all():
                delta = np.zeros(2, dtype=np.float32)
            else:
                delta = self.actor_net(feat).cpu().numpy().reshape(-1)
                if not np.isfinite(delta).all():
                    delta = np.zeros(2, dtype=np.float32)

        if 'prev_action' in state and state['prev_action'] is not None:
            prev_last = np.array(state['prev_action'], dtype=np.float32).reshape(-1)[:2]
        else:
            prev_last = np.array(state['state'][:2], dtype=np.float32).reshape(-1)

        action = (1 - self.alpha) * prev_last + self.alpha * delta
        action = np.clip(action, -1.0, 1.0)
        return action

    def load(self, weight_dir=None):
        """Load pretrained weights from disk."""
        if weight_dir is None:
            weight_dir = getattr(self.opt, 'load_dir', './model_weights/weights')
        device = self.opt.device

        actor_path = os.path.join(weight_dir, 'actor_net.pth')
        feat_path = os.path.join(weight_dir, 'feature_extractor.pth')

        self.actor_net.load_state_dict(
            torch.load(actor_path, map_location=device)
        )
        self.feature_extractor.load_state_dict(
            torch.load(feat_path, map_location=device), strict=True
        )

        print("====================================")
        print("Model has been loaded...")
        print(f"  actor:   {actor_path}")
        print(f"  feature: {feat_path}")
        print("====================================")
