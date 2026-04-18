import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from functools import partial


def gate(x, gate_value):
    return x * gate_value


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_nonlinearity(kind):
    return {
        "relu": F.relu,
        "gelu": F.gelu,
        "swiglu": None,
        "approx_gelu": partial(F.gelu, approximate="tanh"),
        "srelu": lambda x: F.relu(x) ** 2,
        "silu": F.silu,
    }[kind]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


class ProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, non_linearity, dropout, fc_bias=False):
        super().__init__()
        self.swiglu = non_linearity == "swiglu"
        self.dropout = dropout
        self.w1 = nn.Linear(in_dim, out_dim, bias=fc_bias)
        self.w2 = nn.Linear(out_dim, out_dim, bias=fc_bias)
        if self.swiglu:
            self.w3 = nn.Linear(in_dim, out_dim, bias=fc_bias)
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = F.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.w2(hidden)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        theta,
        head_dim,
        max_seqlen=1024,
        scale_factor=1,
        low_freq_factor=1,
        high_freq_factor=32,
        old_context_len=8192,
    ):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        if scale_factor != 1:
            self.low_freq_wavelen = old_context_len / low_freq_factor
            self.high_freq_wavelen = old_context_len / high_freq_factor
            if self.low_freq_wavelen < self.high_freq_wavelen:
                raise ValueError("Invalid RoPE scaling factors")
        self.reset_parameters()

    def reset_parameters(self):
        freqs_cis = self.precompute_freqs_cis(
            dim=self.head_dim,
            end=self.max_seqlen,
            theta=self.theta,
        )
        seq_len, head_dim = freqs_cis.shape[:2]
        freqs_cis = freqs_cis.view(1, seq_len, 1, head_dim, 2, 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def apply_scaling(self, freqs):
        if self.scale_factor == 1:
            return freqs
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < self.high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > self.low_freq_wavelen:
                new_freqs.append(freq / self.scale_factor)
            else:
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / self.scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def precompute_freqs_cis(self, dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = self.apply_scaling(freqs)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        cos, sin = freqs.cos(), freqs.sin()
        return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)

    def forward(self, x, bhle=False):
        if bhle:
            x = x.transpose(1, 2)
        seq_len = x.size(1)
        x_ = x.reshape(*x.shape[:-1], -1, 1, 2)
        x_out = (x_ * self.freqs_cis[:, :seq_len]).sum(5).flatten(3)
        if bhle:
            x_out = x_out.transpose(1, 2)
        return x_out.type_as(x)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding requires an even dim, got {dim}")
        half_dim = dim // 2
        inv_freq = torch.exp(-math.log(theta) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            if x.dim() == 1:
                pos = x
            else:
                pos = torch.arange(x.shape[1], device=x.device)
        emb = torch.einsum("i,j->ij", pos.float(), self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


def pad1d(x, paddings, mode="constant", value=0.0):
    padding_left, padding_right = paddings
    if padding_left < 0 or padding_right < 0:
        raise ValueError(f"Invalid padding values: {paddings}")
    if mode == "reflect":
        length = x.shape[-1]
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class Conv1d(nn.Conv1d):
    def forward(self, x):
        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding))
        return super().forward(x)


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        num_groups=8,
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.activation = nn.SiLU()
        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        num_groups=8,
    ):
        super().__init__()
        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_groups=num_groups,
        )
        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )
        if in_channels != out_channels:
            self.to_out = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)


class Patcher(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        if out_channels % patch_size != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
        )

    def forward(self, x):
        x = self.block(x)
        if self.patch_size == 1:
            return x
        length = x.shape[-1]
        if length % self.patch_size != 0:
            raise ValueError(f"Input length {length} must be divisible by patch_size {self.patch_size}")
        batch, channels, _ = x.shape
        x = x.view(batch, channels, length // self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2).reshape(batch, channels * self.patch_size, length // self.patch_size)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        n_heads,
        n_kv_heads=None,
        norm_eps=1e-5,
        use_qk_norm=False,
        fc_bias=False,
    ):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=fc_bias)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=fc_bias)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=fc_bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=fc_bias)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def reshape_heads(self, x, heads):
        batch, length, channels = x.shape
        x = x.reshape(batch, length, channels // heads, heads)
        return x.permute(0, 3, 1, 2)

    def forward(self, x, cross_x=None, key_padding_mask=None, rope=None):
        xq = self.wq(x)
        if cross_x is not None:
            xk = self.wk(cross_x)
            xv = self.wv(cross_x)
        else:
            xk = self.wk(x)
            xv = self.wv(x)
        xq = self.reshape_heads(xq, self.n_heads)
        xk = self.reshape_heads(xk, self.n_kv_heads)
        xv = self.reshape_heads(xv, self.n_kv_heads)
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
        if rope is not None:
            xq = rope(xq, bhle=True)
            xk = rope(xk, bhle=True)
        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask > 0
            attn_mask = key_padding_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
        batch, _, length, _ = output.shape
        output = output.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        ffn_dim_multiplier,
        multiple_of,
        dropout,
        non_linearity="swiglu",
        fc_bias=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.swiglu = non_linearity == "swiglu"
        if self.swiglu:
            hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=fc_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=fc_bias)
        if self.swiglu:
            self.w3 = nn.Linear(dim, hidden_dim, bias=fc_bias)
        self.non_linearity = get_nonlinearity(non_linearity)

    def forward(self, x):
        hidden1 = self.w1(x)
        if self.swiglu:
            hidden3 = self.w3(x)
            hidden = F.silu(hidden1) * hidden3
        else:
            hidden = self.non_linearity(hidden1)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.w2(hidden)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        dim,
        frequency_embedding_dim,
        non_linearity,
        dropout,
        fc_bias,
        max_period=10000,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_dim
        self.projection = ProjectionLayer(
            in_dim=frequency_embedding_dim,
            out_dim=dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )
        half = frequency_embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t, dim):
        args = t[:, None].float() * self.freqs.to(device=t.device)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t)

    def forward(self, t):
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.projection(x)


class ContextEmbedder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        non_linearity,
        dropout,
        fc_bias,
        norm_eps=1e-5,
        context_norm=False,
    ):
        super().__init__()
        self.context_norm = context_norm
        if context_norm:
            self.norm = RMSNorm(in_dim, norm_eps)
        self.projection = ProjectionLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            non_linearity=non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
        )

    def forward(self, x):
        if x is None:
            return None
        if self.context_norm:
            x = self.norm(x)
        return self.projection(x)


class DiTBlock1D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout=0.0,
        norm_eps=1e-5,
        qk_norm=False,
        fc_bias=False,
        ffn_exp=4,
        ffn_dim_multiplier=1,
        multiple_of=64,
        non_linearity="swiglu",
        no_cross_attention=False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        head_dim = dim // num_heads
        self.attention = Attention(
            dim=dim,
            head_dim=head_dim,
            n_heads=num_heads,
            n_kv_heads=num_heads,
            norm_eps=norm_eps,
            use_qk_norm=qk_norm,
            fc_bias=fc_bias,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=int(ffn_exp * dim),
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
            dropout=dropout,
            non_linearity=non_linearity,
            fc_bias=fc_bias,
        )
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.cross_attention = None
        if not no_cross_attention:
            self.cross_attention = Attention(
                dim=dim,
                head_dim=head_dim,
                n_heads=num_heads,
                n_kv_heads=num_heads,
                norm_eps=norm_eps,
                use_qk_norm=qk_norm,
                fc_bias=fc_bias,
            )
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(self, x, t, cross_x=None, memory_padding_mask=None, rope=None):
        biases = self.scale_shift_table[None] + t.reshape(x.size(0), 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = biases.chunk(6, dim=1)
        h_attn = self.attention(
            modulate(self.attention_norm(x), shift_msa, scale_msa),
            rope=rope,
        )
        h = x + gate(h_attn, gate_msa)
        if self.cross_attention is not None and cross_x is not None:
            h = h + self.cross_attention(
                x=h,
                cross_x=cross_x,
                key_padding_mask=memory_padding_mask,
            )
        h_ff = self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        return h + gate(h_ff, gate_mlp)


class DiTFinalLayer1D(nn.Module):
    def __init__(self, dim, out_channels, dropout=0.0, norm_eps=1e-5, fc_bias=False):
        super().__init__()
        self.dropout = dropout
        self.norm = RMSNorm(dim, norm_eps)
        self.scale_shift_table = nn.Parameter(torch.randn(2, dim) / dim**0.5)
        self.proj = nn.Linear(dim, out_channels, bias=fc_bias)

    def forward(self, x, t_emb):
        shift, scale = (self.scale_shift_table[None] + t_emb[:, None]).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = F.dropout(x, p=self.dropout, training=self.training)
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
        norm_eps=1e-5,
        qk_norm=False,
        fc_bias=True,
        ffn_exp=None,
        ffn_dim_multiplier=1,
        multiple_of=64,
        non_linearity="silu",
        use_rope=False,
        frequency_embedding_dim=None,
        timestep_non_linearity="silu",
        t_block_non_linearity="silu",
        t_block_bias=True,
        context_non_linearity="silu",
        context_embedder_dropout=0.0,
        context_norm=False,
        patch_size=1,
        gradient_checkpointing=False,
    ):
        super().__init__()
        del time_embed_dim
        if model_channels % num_heads != 0:
            raise ValueError(
                f"model_channels={model_channels} must be divisible by num_heads={num_heads}"
            )
        out_channels = in_channels if out_channels is None else out_channels
        ffn_exp = mlp_ratio if ffn_exp is None else ffn_exp
        frequency_embedding_dim = model_channels if frequency_embedding_dim is None else frequency_embedding_dim
        self.model_channels = model_channels
        self.max_length = max_length
        self.dropout = dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.data_proj = nn.Linear(in_channels, model_channels, bias=fc_bias)
        self.x_embedder = Patcher(
            in_channels=model_channels,
            out_channels=model_channels,
            patch_size=patch_size,
        )
        self.rope_embeddings = None
        if use_rope:
            self.rope_embeddings = RotaryEmbedding(
                theta=max(10000, 2 * max_length),
                head_dim=model_channels // num_heads,
                max_seqlen=max_length,
            )
        self.t_embedder = TimestepEmbedder(
            model_channels,
            frequency_embedding_dim,
            non_linearity=timestep_non_linearity,
            dropout=dropout,
            fc_bias=fc_bias,
            max_period=10000,
        )
        self.memory_timestep_embed = SinusoidalEmbedding(model_channels)
        self.t_block_non_linearity = get_nonlinearity(t_block_non_linearity)
        self.t_block = nn.Linear(model_channels, model_channels * 6, bias=t_block_bias)
        self.context_embedder = None
        if context_dim is not None:
            self.context_embedder = ContextEmbedder(
                in_dim=context_dim,
                out_dim=model_channels,
                non_linearity=context_non_linearity,
                dropout=context_embedder_dropout,
                fc_bias=fc_bias,
                norm_eps=norm_eps,
                context_norm=context_norm,
            )
        self.blocks = nn.ModuleList(
            [
                DiTBlock1D(
                    dim=model_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    fc_bias=fc_bias,
                    ffn_exp=ffn_exp,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    multiple_of=multiple_of,
                    non_linearity=non_linearity,
                    no_cross_attention=context_dim is None,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = DiTFinalLayer1D(
            dim=model_channels,
            out_channels=out_channels,
            dropout=dropout,
            norm_eps=norm_eps,
            fc_bias=fc_bias,
        )

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

    def _forward_block(self, block, h, t_block, memory, mask):
        return block(
            x=h,
            t=t_block,
            cross_x=memory,
            memory_padding_mask=mask,
            rope=self.rope_embeddings,
        )

    def forward(self, x, timesteps, context_list=None, y=None, context_attn_mask_list=None):
        del y
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected input rank {x.dim()}; expected 3")
        h = x.transpose(1, 2)
        h = self.data_proj(h)
        h = self.x_embedder(h.transpose(1, 2)).transpose(1, 2)
        h = F.dropout(h, p=self.dropout, training=self.training)
        t_emb = self.t_embedder(timesteps)
        t_block = self.t_block(self.t_block_non_linearity(t_emb))
        context, mask = self._merge_context(context_list, context_attn_mask_list)
        memory = self.context_embedder(context) if self.context_embedder is not None else None
        if memory is not None:
            memory = memory + self.memory_timestep_embed(timesteps, pos=timesteps).to(memory).unsqueeze(1)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                if memory is None:
                    h = checkpoint(
                        lambda h_, t_: self._forward_block(block, h_, t_, None, mask),
                        h,
                        t_block,
                        use_reentrant=False,
                    )
                else:
                    h = checkpoint(
                        lambda h_, t_, memory_: self._forward_block(block, h_, t_, memory_, mask),
                        h,
                        t_block,
                        memory,
                        use_reentrant=False,
                    )
            else:
                h = self._forward_block(block, h, t_block, memory, mask)
        out = self.final_layer(h, t_emb)
        return out.transpose(1, 2)
