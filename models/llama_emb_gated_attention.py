import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    norm_eps: float
    rope_theta: float
    max_sequence_length: int
    gated_attention_type: str


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        return self.weight * x

    def reset_parameters(self):
        nn.init.ones_(self.weight)


def compute_default_rope_parameters(theta, head_dim, max_sequence_length):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)) # shape (head_dim/2,)
    position_ids = torch.arange(max_sequence_length, dtype=torch.float32) # shape (max_sequence_length,)
    freqs = torch.outer(position_ids, inv_freq) # shape (max_sequence_length, head_dim/2)
    
    emb = torch.cat((freqs, freqs), dim=-1)[None, None, :, :] # shape (1, 1, max_sequence_length, head_dim)
    cos = emb.cos() # shape (1, 1, max_sequence_length, head_dim)
    sin = emb.sin() # shape (1, 1, max_sequence_length, head_dim)
    return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_ = q.float() # shape (batch_size, num_heads, num_tokens, head_dim)
    k_ = k.float() # shape (batch_size, num_kv_heads, num_tokens, head_dim)

    num_tokens = q_.shape[-2]
    cos = cos[:, :, :num_tokens, :]
    sin = sin[:, :, :num_tokens, :]

    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, gated_attention_type):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gated_attention_type = gated_attention_type

        self.wq = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        if gated_attention_type == "standard":
            self.wg = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        elif gated_attention_type == "head":
            self.wg = nn.Linear(hidden_size, num_heads, bias=False)
        elif gated_attention_type == "token":
            self.wg = nn.Linear(hidden_size, 1, bias=False)
        elif gated_attention_type is not None:
            raise ValueError(f"Unsupported gated_attention_type: {gated_attention_type}")

    def forward(self, x, emb, position_embeddings, return_attention_map=False):
        batch_size, sequence_length, _ = x.shape
        q = self.wq(x).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        k = self.wk(x).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2) # (B, num_kv_heads, N, head_dim)
        v = self.wv(x).view(batch_size, sequence_length, -1, self.head_dim).transpose(1, 2) # (B, num_kv_heads, N, head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if return_attention_map:
            # For returning attention matrix, we need to compute it manually instead of using F.scaled_dot_product_attention, since the latter does not return attention matrix.
            # Handle GQA by repeating keys and values for each head if num_kv_heads < num_heads
            num_kv_groups = self.num_heads // self.num_kv_heads
            if num_kv_groups > 1:
                k = k[:, :, None, :, :].expand(-1, -1, num_kv_groups, -1, -1).reshape(batch_size, self.num_heads, sequence_length, self.head_dim)
                v = v[:, :, None, :, :].expand(-1, -1, num_kv_groups, -1, -1).reshape(batch_size, self.num_heads, sequence_length, self.head_dim)
            attention_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            causal_mask = torch.triu(torch.full((sequence_length, sequence_length), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
            attention_score = attention_score + causal_mask
            attention_map = torch.softmax(attention_score, dim=-1, dtype=torch.float32).type_as(q)
            a = torch.matmul(attention_map, v).transpose(1, 2).reshape(batch_size, sequence_length, -1).to(q.dtype)
        else:
            with nn.attention.sdpa_kernel([nn.attention.SDPBackend.CUDNN_ATTENTION, nn.attention.SDPBackend.FLASH_ATTENTION], set_priority=True):
                a = F.scaled_dot_product_attention(
                    q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                    is_causal=True,
                    enable_gqa=True
                ).transpose(1, 2).reshape(batch_size, sequence_length, -1).to(q.dtype)
            attention_map = None

        if self.gated_attention_type == "standard":
            g = self.wg(emb) # shape (batch_size, sequence_length, num_heads * head_dim)
            a = a * torch.sigmoid(g)
        elif self.gated_attention_type == "head":
            g = self.wg(emb).view(batch_size, sequence_length, self.num_heads, 1) # shape (batch_size, sequence_length, num_heads, 1)
            a = a.view(batch_size, sequence_length, self.num_heads, self.head_dim) * torch.sigmoid(g)
            a = a.view(batch_size, sequence_length, -1)
        elif self.gated_attention_type == "token":
            g = self.wg(emb).view(batch_size, sequence_length, 1) # shape (batch_size, sequence_length, 1)
            a = a * torch.sigmoid(g)
        else:
            assert self.gated_attention_type is None, f"Unsupported gated_attention_type: {self.gated_attention_type}"

        return self.wo(a), attention_map

    
    def reset_parameters(self):
        nn.init.normal_(self.wq.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wk.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wv.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)
    

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def reset_parameters(self):
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)


class LlamaBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, norm_eps, gated_attention_type):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            gated_attention_type=gated_attention_type,
        )

        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x, emb, position_embeddings, return_attention_map):
        attention_output, attention_map = self.attn(self.attn_norm(x), emb, position_embeddings, return_attention_map)
        h = x + attention_output
        x = h + self.ffn(self.ffn_norm(h))
        return x, h, attention_map

    def reset_parameters(self):
        self.attn_norm.reset_parameters()
        self.attn.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        cos, sin = compute_default_rope_parameters(
            theta=config.rope_theta,
            head_dim=config.head_dim,
            max_sequence_length=config.max_sequence_length,
        )
        self.register_buffer("position_embeddings_cos", cos, persistent=False)
        self.register_buffer("position_embeddings_sin", sin, persistent=False)

        self.layers = nn.ModuleList([
            LlamaBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                norm_eps=config.norm_eps,
                gated_attention_type=config.gated_attention_type,
            ) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, record_mode=False):
        emb = self.embedding(x)
        h = emb

        if record_mode:
            all_attention_maps = []
            all_hidden_representations = [h]

        position_embeddings = (self.position_embeddings_cos, self.position_embeddings_sin)
        for layer in self.layers:
            h, hidden_representation, attention_map = layer(h, emb, position_embeddings, return_attention_map=record_mode)
            if record_mode:
                all_attention_maps.append(attention_map)
                all_hidden_representations.append(hidden_representation)
                all_hidden_representations.append(h)

        if record_mode:
            return self.head(self.norm(h)), all_attention_maps, all_hidden_representations
        else:
            return self.head(self.norm(h))
    
    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

        cos, sin = compute_default_rope_parameters(
            theta=self.config.rope_theta,
            head_dim=self.config.head_dim,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.position_embeddings_cos.copy_(cos)
        self.position_embeddings_sin.copy_(sin)

        for layer in self.layers:
            layer.reset_parameters()
        
        self.norm.reset_parameters()
        nn.init.normal_(self.head.weight, mean=0.0, std=self.config.hidden_size**-0.5)
        