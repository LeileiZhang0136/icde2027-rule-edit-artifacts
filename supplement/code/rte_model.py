"""Architecture-aligned RTE modules for Qwen2.5 (Transformers 4.57).

RuleEdit uses a T5 amortizer, a T5 query encoder, cross-attention over a
RuleBank, and layer-wise KV soft prefixes. Qwen2.5 is decoder-only and uses
grouped-query attention, so its prefix tensor is [layer, K/V, batch, KV-head,
prefix-token, head-dim]. This is an adaptation, not the authors' code.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, MT5EncoderModel, MT5Model, T5EncoderModel, T5Model
from transformers.cache_utils import DynamicCache


class AggregationBlock(nn.Module):
    def __init__(self, width: int, heads: int = 8) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feed_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
        query = query + attended
        return query + self.feed_forward(self.feed_norm(query))


class RTEPrefix(nn.Module):
    def __init__(self, qwen_config, t5_model: str, *, prefix_tokens: int = 24, cross_attention_blocks: int = 4, local_files_only: bool = True) -> None:
        super().__init__()
        t5_config = AutoConfig.from_pretrained(t5_model, local_files_only=local_files_only)
        if t5_config.model_type == "t5":
            amortizer_class, encoder_class = T5Model, T5EncoderModel
        elif t5_config.model_type == "mt5":
            amortizer_class, encoder_class = MT5Model, MT5EncoderModel
        else:
            raise ValueError(f"Only T5/mT5 checkpoints are supported, got {t5_config.model_type}")
        self.amortizer = amortizer_class.from_pretrained(t5_model, local_files_only=local_files_only)
        self.query_encoder = encoder_class.from_pretrained(t5_model, local_files_only=local_files_only)
        width = self.amortizer.config.d_model
        if self.query_encoder.config.d_model != width:
            raise ValueError("T5 encoder/amortizer dimensions differ")
        self.prefix_tokens = prefix_tokens
        self.layers = qwen_config.num_hidden_layers
        self.kv_heads = qwen_config.num_key_value_heads
        self.head_dim = getattr(qwen_config, "head_dim", qwen_config.hidden_size // qwen_config.num_attention_heads)
        self.rule_decoder_slots = nn.Parameter(torch.randn(prefix_tokens, width) * 0.02)
        self.query_slots = nn.Parameter(torch.randn(prefix_tokens, width) * 0.02)
        self.aggregator = nn.ModuleList(AggregationBlock(width) for _ in range(cross_attention_blocks))
        output_width = self.layers * 2 * self.kv_heads * self.head_dim
        # Separate two-layer MLP for each virtual token, as specified in the paper.
        self.kv_heads_by_slot = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 256), nn.Tanh(), nn.Linear(256, output_width))
            for _ in range(prefix_tokens)
        )

    def encode_rulebank(self, rule_input_ids: torch.Tensor, rule_attention_mask: torch.Tensor) -> torch.Tensor:
        count = rule_input_ids.shape[0]
        decoder_inputs = self.rule_decoder_slots.unsqueeze(0).expand(count, -1, -1)
        result = self.amortizer(
            input_ids=rule_input_ids,
            attention_mask=rule_attention_mask,
            decoder_inputs_embeds=decoder_inputs,
            decoder_attention_mask=torch.ones(count, self.prefix_tokens, device=rule_input_ids.device, dtype=rule_attention_mask.dtype),
            use_cache=False,
            return_dict=True,
        )
        return result.last_hidden_state

    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor, rulebank: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if rulebank.ndim != 3 or rulebank.shape[1] != self.prefix_tokens:
            raise ValueError("RuleBank must have shape [rules, prefix_tokens, T5_width]")
        encoded = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask, return_dict=True).last_hidden_state
        weights = query_attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        query = pooled.unsqueeze(1) + self.query_slots.unsqueeze(0)
        memory = rulebank.reshape(1, -1, rulebank.shape[-1]).expand(query.shape[0], -1, -1)
        for block in self.aggregator:
            query = block(query, memory)
        by_slot = torch.stack([head(query[:, i]) for i, head in enumerate(self.kv_heads_by_slot)], dim=1)
        # [B, P, L*2*KV*D] -> [L, 2, B, KV, P, D]
        shaped = by_slot.reshape(query.shape[0], self.prefix_tokens, self.layers, 2, self.kv_heads, self.head_dim)
        shaped = shaped.permute(2, 3, 0, 4, 1, 5).contiguous()
        return tuple((shaped[layer, 0], shaped[layer, 1]) for layer in range(self.layers))


def prefix_cache(prefix: tuple[tuple[torch.Tensor, torch.Tensor], ...], *, dtype: torch.dtype) -> DynamicCache:
    pairs = [(key.to(dtype=dtype), value.to(dtype=dtype)) for key, value in prefix]
    if not pairs or any(key.shape != value.shape for key, value in pairs):
        raise ValueError("Invalid layer-wise KV prefix")
    return DynamicCache(ddp_cache_data=pairs)


def trainable_parameter_report(qwen: nn.Module, rte: nn.Module) -> dict[str, int]:
    return {
        "qwen_trainable": sum(p.numel() for p in qwen.parameters() if p.requires_grad),
        "rte_trainable": sum(p.numel() for p in rte.parameters() if p.requires_grad),
    }
