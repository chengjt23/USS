import torch
import torch.nn as nn
import torch.nn.functional as F

from models.flowsep.attention import CrossAttention, FeedForward
from utils.diffusion import timestep_embedding, zero_module


class DiTBlock1D(nn.Module):
    def __init__(
        self,
        dim,
        time_embed_dim,
        num_heads,
        dim_head,
        context_dim=None,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = CrossAttention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, mult=mlp_ratio, glu=True, dropout=dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, dim * 9))

    def _modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, t_emb, context=None, mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = self.modulation(t_emb).chunk(9, dim=1)
        h = self._modulate(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.self_attn(h)
        if context is not None:
            h = self._modulate(self.norm2(x), shift2, scale2)
            x = x + gate2.unsqueeze(1) * self.cross_attn(h, context=context, mask=mask)
        h = self._modulate(self.norm3(x), shift3, scale3)
        x = x + gate3.unsqueeze(1) * self.ff(h)
        return x


class DiTFinalLayer1D(nn.Module):
    def __init__(self, dim, time_embed_dim, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, dim * 2))
        self.proj = zero_module(nn.Linear(dim, out_channels))

    def forward(self, x, t_emb):
        shift, scale = self.modulation(t_emb).chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj(x)


class DiT1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        model_channels=512,
        num_layers=8,
        num_heads=8,
        context_dim=1024,
        mlp_ratio=4.0,
        dropout=0.0,
        max_length=104,
        time_embed_dim=None,
    ):
        super().__init__()
        if model_channels % num_heads != 0:
            raise ValueError(
                f"model_channels={model_channels} must be divisible by num_heads={num_heads}"
            )
        out_channels = in_channels if out_channels is None else out_channels
        time_embed_dim = model_channels * 4 if time_embed_dim is None else time_embed_dim
        dim_head = model_channels // num_heads
        self.model_channels = model_channels
        self.max_length = max_length
        self.in_proj = nn.Conv1d(in_channels, model_channels, kernel_size=1)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    dim=model_channels,
                    time_embed_dim=time_embed_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    context_dim=context_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer1D(model_channels, time_embed_dim, out_channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, model_channels))
        nn.init.normal_(self.pos_embed, std=0.02)

    def _get_pos_embed(self, length, dtype, device):
        if length == self.max_length:
            return self.pos_embed.to(device=device, dtype=dtype)
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=length, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def _merge_context(self, context_list=None, mask_list=None):
        if not context_list:
            return None, None
        contexts = []
        masks = []
        use_mask = True
        for idx, context in enumerate(context_list):
            if context is None:
                continue
            contexts.append(context)
            if mask_list is None or idx >= len(mask_list) or mask_list[idx] is None:
                use_mask = False
            else:
                masks.append(mask_list[idx])
        if not contexts:
            return None, None
        context = torch.cat(contexts, dim=1) if len(contexts) > 1 else contexts[0]
        if not use_mask:
            return context, None
        mask = torch.cat(masks, dim=1) if len(masks) > 1 else masks[0]
        return context, mask

    def forward(self, x, timesteps, context_list=None, y=None, context_attn_mask_list=None):
        del y
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected input rank {x.dim()}; expected 3")
        h = self.in_proj(x).transpose(1, 2)
        h = h + self._get_pos_embed(h.shape[1], h.dtype, h.device)
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        context, mask = self._merge_context(context_list, context_attn_mask_list)
        for block in self.blocks:
            h = block(h, t_emb, context=context, mask=mask)
        out = self.final_layer(h, t_emb)
        return out.transpose(1, 2)
