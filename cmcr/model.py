"""Cross-modal moment-commutator rotations of post-RoPE queries and keys."""
from __future__ import annotations
import math, types
import torch
from torch import nn
from torch.nn import functional as F

def skew(x):
    return 0.5 * (x - x.transpose(-1, -2))

def cayley(a):
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    return torch.linalg.solve(eye - a, eye + a)

def bounded(x, floor=1.0):
    return x / x.norm(dim=(-1, -2), keepdim=True).clamp_min(floor)

def moments(z, mask):
    weights = mask.float()
    count = weights.sum(1, keepdim=True).clamp_min(1)
    mu = (z * weights[..., None]).sum(1) / count
    centered = (z - mu[:, None, :]) * weights[..., None]
    cov = centered.transpose(-1, -2) @ centered / count[..., None]
    cov = cov / cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-06)
    return (F.normalize(mu, dim=-1, eps=1e-06), cov)

def generators(mu_t, c_t, mu_v, c_v):
    mean = mu_t[..., None] * mu_v[:, None, :] - mu_v[..., None] * mu_t[:, None, :]
    comm = c_t @ c_v - c_v @ c_t
    return (bounded(mean), bounded(comm, 0.02))

def subspace_rotate(x, u, r):
    projected = torch.einsum('bhsd,hdr->bhsr', x.float(), u)
    eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
    edited = torch.einsum('bhsr,bhrk->bhsk', projected, r - eye)
    return x.float() + torch.einsum('bhsr,hdr->bhsd', edited, u)

class CommutatorRotator(nn.Module):

    def __init__(self, hidden_size=2560, head_dim=128, kv_heads=8, levels=3, rank=16, variant='cmcr'):
        super().__init__()
        if variant not in ('cmcr', 'static', 'no_comm'):
            raise ValueError(f'Unsupported CMCR variant: {variant}')
        self.variant = variant
        self.intervention = 'none'
        self.rank = rank
        self.kv_heads = kv_heads
        self.levels = levels
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(levels)])
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.bases = nn.ParameterList([nn.Parameter(torch.randn(kv_heads, head_dim, rank) / math.sqrt(head_dim)) for _ in range(levels)])
        self.static = nn.ParameterList([nn.Parameter(torch.randn(kv_heads, rank, rank) * 0.01) for _ in range(levels)])
        self.controllers = nn.ModuleList([nn.Sequential(nn.Linear(6 * rank, 64), nn.GELU(), nn.Linear(64, kv_heads * 2)) for _ in range(levels)])
        for controller in self.controllers:
            nn.init.normal_(controller[-1].weight, std=0.01)
            nn.init.constant_(controller[-1].bias, 0.5)
        self.strength = nn.Parameter(torch.full((levels, kv_heads), -1.0))
        self.masks = None
        self.traces = {}

    def frames(self, hidden, level):
        text, visual = self.masks
        z = torch.tanh(self.down(self.norms[level](hidden.float())))
        mt, ct = moments(z, text)
        mv, cv = moments(z, visual)
        first, second = generators(mt, ct, mv, cv)
        descriptor = torch.cat([mt, mv, mt * mv, (mt - mv).abs(), ct.diagonal(dim1=-2, dim2=-1), cv.diagonal(dim1=-2, dim2=-1)], -1)
        coefficients = self.controllers[level](descriptor).sigmoid().view(len(z), self.kv_heads, 2)
        constant = skew(self.static[level])[None].expand(len(z), -1, -1, -1)
        if self.variant == 'static':
            raw = constant
        else:
            use_comm = self.variant == 'cmcr' and self.intervention != 'no_comm'
            raw = constant + coefficients[:, :, 0, None, None] * first[:, None]
            if use_comm:
                raw = raw + coefficients[:, :, 1, None, None] * second[:, None]
        a = 0.35 * self.strength[level].sigmoid()[None, :, None, None] * bounded(raw)
        if self.intervention == 'reverse_rotation':
            a = -a
        rotation = cayley(a)
        u = torch.linalg.qr(self.bases[level], mode='reduced').Q
        eye = torch.eye(self.rank, device=hidden.device, dtype=torch.float32)
        self.traces[level] = {'energy': (rotation - eye).square().mean(), 'comm_norm': second.norm(dim=(-1, -2)).mean().detach(), 'angle_proxy': (rotation - eye).norm(dim=(-1, -2)).mean().detach(), 'mean_gate': coefficients[:, :, 0].mean().detach(), 'comm_gate': coefficients[:, :, 1].mean().detach()}
        return (u, rotation)

    def forward(self, hidden, q, k, level):
        if self.intervention == 'bypass':
            self.traces[level] = {'energy': hidden.float().sum() * 0}
            return (q, k)
        assert self.masks is not None
        u, r = self.frames(hidden, level)

        def edit(x):
            repeats = x.shape[1] // self.kv_heads
            assert repeats * self.kv_heads == x.shape[1]
            basis = u.repeat_interleave(repeats, 0)
            frame = r.repeat_interleave(repeats, 1)
            pos = subspace_rotate(x, basis, frame)
            inverse = frame if self.intervention == 'common_frame' else frame.transpose(-1, -2)
            neg = subspace_rotate(x, basis, inverse)
            out = torch.where(self.masks[1][:, None, :, None], neg, pos)
            return out.to(x.dtype)
        return (edit(q), edit(k))

def install_attention(attn, rotator, level):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs):
        if past_key_values is not None or kwargs.get('past_key_value') is not None:
            raise ValueError('CMCR is a full-input discriminative classifier; KV cache unsupported')
        shape = hidden_states.shape[:-1]
        hshape = (*shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hshape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        q, k = rotator(hidden_states, q, k, level)
        interface = eager_attention_forward
        if self.config._attn_implementation != 'eager':
            interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, weights = interface(self, q, k, v, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs)
        return (self.o_proj(out.reshape(*shape, -1).contiguous()), weights)
    attn.forward = types.MethodType(forward, attn)

class CMCRModel(nn.Module):

    def __init__(self, backbone, layers=(11, 23, 31), rank=16, variant='cmcr'):
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        base = backbone.get_base_model() if hasattr(backbone, 'get_base_model') else backbone
        c = base.config.text_config
        self.rotator = CommutatorRotator(c.hidden_size, c.head_dim, c.num_key_value_heads, len(layers), rank, variant)
        self.head = nn.Sequential(nn.LayerNorm(c.hidden_size * 2), nn.Linear(c.hidden_size * 2, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2))
        for level, layer in enumerate(layers):
            install_attention(base.model.language_model.layers[layer].self_attn, self.rotator, level)

    def forward(self, text_mask, visual_mask, **inputs):
        self.rotator.masks = (text_mask, visual_mask)
        self.rotator.traces = {}
        base = self.backbone.get_base_model() if hasattr(self.backbone, 'get_base_model') else self.backbone
        h = base.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        mask = text_mask.to(h.dtype)[..., None]
        mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
        last = h[torch.arange(len(h), device=h.device), inputs['attention_mask'].sum(-1) - 1]
        logits = self.head(torch.cat([mean, last], -1).float())
        energy = torch.stack([self.rotator.traces[i]['energy'] for i in range(len(self.layers))]).mean()
        stats = {key: torch.stack([t[key] for t in self.rotator.traces.values() if key in t]).mean() for key in ['comm_norm', 'angle_proxy', 'mean_gate', 'comm_gate'] if any((key in t for t in self.rotator.traces.values()))}
        return {'logits': logits, 'energy': energy, 'stats': stats}

    def custom_state(self):
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if not n.startswith('backbone.')}

def build_model(config):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(config['base_model'], dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    backbone = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=0.05, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], exclude_modules='.*visual.*'))
    torch.manual_seed(config['seed'] + 900)
    return CMCRModel(backbone, config['layers'], config['rank'], config['variant'])
