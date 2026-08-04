import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)

class MLP(nn.Module):

    def __init__(self, dim, inter_multi=4, bias=False):
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x, *args):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


def run_prope_attention_by_room(q, k, v, attn_fn, viewmats, room_ids, num_views):
    if room_ids is None or num_views <= 1:
        return attn_fn(q, k, v, viewmats=viewmats, Ks=None)

    if room_ids.dim() == 1:
        room_ids = room_ids.unsqueeze(0)

    batch, num_heads, seqlen, head_dim = q.shape
    if room_ids.shape[0] != batch or room_ids.shape[1] != num_views:
        raise ValueError(
            f"room_ids shape {tuple(room_ids.shape)} does not match batch/views {(batch, num_views)}"
        )
    if seqlen % num_views != 0:
        raise ValueError(f"seqlen {seqlen} is not divisible by num_views {num_views}")

    if torch.all(room_ids == room_ids[:, :1]):
        return attn_fn(q, k, v, viewmats=viewmats, Ks=None)

    tokens_per_view = seqlen // num_views
    q_by_view = q.reshape(batch, num_heads, num_views, tokens_per_view, head_dim)
    k_by_view = k.reshape(batch, num_heads, num_views, tokens_per_view, head_dim)
    v_by_view = v.reshape(batch, num_heads, num_views, tokens_per_view, head_dim)
    batch_outputs = []
    for b_idx in range(batch):
        room_ids_b = room_ids[b_idx]
        sample_out = None
        for room_id in torch.unique(room_ids_b):
            view_indices = torch.nonzero(room_ids_b == room_id, as_tuple=False).flatten()
            q_room = q_by_view[b_idx:b_idx + 1, :, view_indices].contiguous().reshape(1, num_heads, -1, head_dim)
            k_room = k_by_view[b_idx:b_idx + 1, :, view_indices].contiguous().reshape(1, num_heads, -1, head_dim)
            v_room = v_by_view[b_idx:b_idx + 1, :, view_indices].contiguous().reshape(1, num_heads, -1, head_dim)
            room_out = attn_fn(
                q_room,
                k_room,
                v_room,
                viewmats=viewmats[b_idx:b_idx + 1, view_indices],
                Ks=None,
            )
            room_out = room_out.reshape(1, num_heads, view_indices.numel(), tokens_per_view, head_dim)
            if sample_out is None:
                sample_out = torch.empty(
                    (1, num_heads, num_views, tokens_per_view, head_dim),
                    device=room_out.device,
                    dtype=room_out.dtype,
                )
            sample_out[:, :, view_indices] = room_out
        batch_outputs.append(sample_out)

    out = torch.cat(batch_outputs, dim=0)
    return out.reshape(batch, num_heads, seqlen, head_dim)

class SelfAttention(nn.Module):
    """
    Self-attention layer
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        use_qk_norm=True,
        causal=False,
        bias=False,
    ):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x, prope, stage, *args):
        """
        x: (b, l, d)
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if stage == 1:
            x = F.scaled_dot_product_attention(q, k, v)

        elif stage == 2:
            if prope:
                w2c = args[0]["w2c"]
                attn_fn = args[0]["attn2"]
                room_ids = args[0].get("input_room_ids")

                x = run_prope_attention_by_room(
                    q,
                    k,
                    v,
                    attn_fn=attn_fn,
                    viewmats=w2c,
                    room_ids=room_ids,
                    num_views=args[0]["num_input_views"],
                )
            else:
                attn_fn = args[0]["attn2"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,
                )
        elif stage == 3:
            if prope:
                w2c = args[0]["w2c"]
                attn_fn = args[0]["attn3"]
                room_ids = args[0].get("input_room_ids")

                x = run_prope_attention_by_room(
                    q,
                    k,
                    v,
                    attn_fn=attn_fn,
                    viewmats=w2c,
                    room_ids=room_ids,
                    num_views=args[0]["num_input_views"],
                )
            else:
                attn_fn = args[0]["attn3"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,
                )
        x = rearrange(x, "b nh l dh -> b l (nh dh)")

        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias, head_dim, inter_multi, use_qk_norm):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.attn = SelfAttention(dim=dim, bias=bias, head_dim=head_dim, use_qk_norm=use_qk_norm)

        self.ln2 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.mlp = MLP(dim=dim, bias=bias, inter_multi=inter_multi)

    def forward(self, x, prope, stage, info):
        x = x + self.attn(self.ln1(x), prope, stage, info)
        x = x + self.mlp(self.ln2(x))
        return x
