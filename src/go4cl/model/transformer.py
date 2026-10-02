"""Decoder-only causal Transformer for single-slot modular addition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from go4cl.constants import CONTEXT_LENGTH, NUM_OUTPUT_CLASSES, VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_classes: int = NUM_OUTPUT_CLASSES
    context_length: int = CONTEXT_LENGTH
    n_layers: int = 3
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    dropout: float = 0.0
    activation: str = "relu"  # relu | gelu
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"unsupported activation: {self.activation}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length))
        self.register_buffer("mask", mask.view(1, 1, cfg.context_length, cfg.context_length))

    def forward(
        self, x: torch.Tensor, *, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out, att, _ = self.forward_detailed(x)
        if return_attn:
            return out, att
        return out

    def forward_detailed(
        self,
        x: torch.Tensor,
        *,
        ablate_heads: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention forward with optional head ablation.

        Returns ``(out, attn_probs, y_pre_out)`` where ``y_pre_out`` is the
        concatenated head outputs [B, T, D] before the output projection.
        Heads listed in ``ablate_heads`` are zeroed in that concatenation.
        """
        b, t, c = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, T, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.d_head**-0.5)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y_heads = att @ v  # [B, H, T, Dh]
        if ablate_heads:
            y_heads = y_heads.clone()
            for h in ablate_heads:
                hi = int(h)
                if 0 <= hi < self.n_heads:
                    y_heads[:, hi] = 0
        y = y_heads.transpose(1, 2).contiguous().view(b, t, c)
        out = self.resid_drop(self.out(y))
        return out, att, y


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.fc2 = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = F.relu if cfg.activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, *, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attn:
            attn_out, att, _ = self.attn.forward_detailed(self.ln1(x))
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, att
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ModularTransformer(nn.Module):
    """
    Decoder-only causal Transformer.

    Prediction is taken from the final query position (index -1).
    Input embedding and output head are not tied by default.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.context_length, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.n_classes, bias=True)
        if c.tie_embeddings:
            raise ValueError("input/output tying is disabled for this setup")
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        b, t = tokens.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence length {t} > context {self.cfg.context_length}")
        pos = torch.arange(t, device=tokens.device).unsqueeze(0)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :])  # query position
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

    @torch.no_grad()
    def forward_with_cache(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass that returns residual stream + attention caches.

        Cache keys:
          - tok_emb: token embeddings before pos/drop  [B, T, D]
          - resid_pre: residual entering each block    list[L] of [B, T, D]
          - resid_mid: residual after attention        list[L] of [B, T, D]
          - resid_post: residual after MLP             list[L] of [B, T, D]
          - attn: attention probs                      list[L] of [B, H, T, T]
          - query_resid: ln_f(query position)          [B, D]
          - logits: class logits                       [B, C]
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [B, T], got {tuple(tokens.shape)}")
        b, t = tokens.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence length {t} > context {self.cfg.context_length}")

        was_training = self.training
        self.eval()

        pos = torch.arange(t, device=tokens.device).unsqueeze(0)
        tok = self.tok_emb(tokens)
        x = self.drop(tok + self.pos_emb(pos))

        resid_pre: list[torch.Tensor] = []
        resid_mid: list[torch.Tensor] = []
        resid_post: list[torch.Tensor] = []
        attns: list[torch.Tensor] = []

        for block in self.blocks:
            resid_pre.append(x)
            attn_out, att = block.attn(block.ln1(x), return_attn=True)
            mid = x + attn_out
            resid_mid.append(mid)
            post = mid + block.mlp(block.ln2(mid))
            resid_post.append(post)
            attns.append(att)
            x = post

        x = self.ln_f(x)
        query_resid = x[:, -1, :]
        logits = self.head(query_resid)
        out: dict[str, Any] = {
            "tok_emb": tok,
            "resid_pre": resid_pre,
            "resid_mid": resid_mid,
            "resid_post": resid_post,
            "attn": attns,
            "query_resid": query_resid,
            "logits": logits,
        }
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)

        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def continue_from_layer(
        self,
        resid_post: torch.Tensor,
        *,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Continue from residual *after* ``layer_idx``, through later blocks + head.

        ``resid_post`` shape [B, T, D] must match the post-MLP residual of
        ``self.blocks[layer_idx]``. Used for mid-layer interventions.
        """
        if not (0 <= layer_idx < len(self.blocks)):
            raise ValueError(f"layer_idx={layer_idx} out of range")
        was_training = self.training
        self.eval()
        x = resid_post
        for block in self.blocks[layer_idx + 1 :]:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :])
        if was_training:
            self.train()
        return {"logits": logits, "query_resid": x[:, -1, :]}

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
