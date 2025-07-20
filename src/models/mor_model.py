"""
Mixture-of-Recursions (MoR) Implementation

This module implements the core MoR architecture as described in:
"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoRConfig:
    """Configuration for Mixture-of-Recursions model."""

    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024

    # MoR specific parameters
    max_recursion_depth: int = 4
    min_recursion_depth: int = 1
    router_hidden_size: int = 128
    router_dropout: float = 0.1

    # Training parameters
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # KV sharing variant
    use_kv_sharing: bool = False

    # Efficiency settings
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False


class RecursiveRouter(nn.Module):
    """Lightweight router for determining recursion depth per token."""

    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        # Router network
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.router_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.router_dropout),
            nn.Linear(config.router_hidden_size, config.max_recursion_depth),
        )

        # Temperature for Gumbel softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, hidden_states: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training: Whether in training mode

        Returns:
            recursion_depths: [batch_size, seq_len] - depth assignment per token
            router_logits: [batch_size, seq_len, max_depth] - raw router outputs
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get router logits
        router_logits = self.router(hidden_states)  # [B, L, max_depth]

        if training:
            # Use Gumbel softmax for differentiable sampling
            recursion_probs = F.gumbel_softmax(
                router_logits, tau=self.temperature, hard=True
            )
            recursion_depths = (
                torch.argmax(recursion_probs, dim=-1) + self.config.min_recursion_depth
            )
        else:
            # Use argmax for inference
            recursion_depths = (
                torch.argmax(router_logits, dim=-1) + self.config.min_recursion_depth
            )

        return recursion_depths, router_logits


class RecursiveTransformerLayer(nn.Module):
    """Shared transformer layer that can be applied recursively."""

    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len, seq_len]
            key_value_cache: Cached (key, value) tensors
            use_cache: Whether to return key-value cache

        Returns:
            hidden_states: Updated hidden states
            key_value_cache: Updated cache if use_cache=True
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        if key_value_cache is not None:
            # Use cached key-value pairs (for KV sharing variant)
            cached_key, cached_value = key_value_cache
            attn_output, attn_weights = self.attention(
                hidden_states,
                cached_key,
                cached_value,
                attn_mask=attention_mask,
                need_weights=False,
            )
            new_cache = key_value_cache if use_cache else None
        else:
            # Standard self-attention
            attn_output, attn_weights = self.attention(
                hidden_states,
                hidden_states,
                hidden_states,
                attn_mask=attention_mask,
                need_weights=False,
            )

            if use_cache:
                # Create new cache (simplified - in practice would extract from attention module)
                new_cache = (hidden_states, hidden_states)
            else:
                new_cache = None

        hidden_states = residual + attn_output

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return hidden_states, new_cache


class MixtureOfRecursions(nn.Module):
    """Mixture-of-Recursions Transformer Model."""

    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

        # Shared recursive transformer layers
        self.recursive_layers = nn.ModuleList(
            [RecursiveTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Router for recursion depth assignment
        self.router = RecursiveRouter(config)

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
                - logits: [batch_size, seq_len, vocab_size]
                - recursion_depths: [batch_size, seq_len]
                - router_loss: Scalar tensor
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        hidden_states = self.dropout(token_embeds + pos_embeds)

        # Get recursion depths from router
        recursion_depths, router_logits = self.router(
            hidden_states, training=self.training
        )

        # Apply recursive layers with adaptive computation
        kv_cache = None
        for depth in range(1, self.config.max_recursion_depth + 1):
            # Create mask for tokens that should be processed at this depth
            depth_mask = (recursion_depths >= depth).float()  # [B, L]

            if depth_mask.sum() == 0:
                continue  # No tokens need processing at this depth

            # Selective attention: only process active tokens
            active_indices = depth_mask.nonzero(as_tuple=True)
            if len(active_indices[0]) == 0:
                continue

            # For simplicity, process all tokens but mask inactive ones
            # In practice, you'd want more sophisticated selective computation
            layer_idx = (depth - 1) % len(self.recursive_layers)
            layer = self.recursive_layers[layer_idx]

            # Apply layer with optional KV sharing
            use_cache = depth == 1 and self.config.use_kv_sharing
            hidden_states, kv_cache = layer(
                hidden_states,
                attention_mask=self._create_attention_mask(attention_mask, depth_mask),
                key_value_cache=(
                    kv_cache if depth > 1 and self.config.use_kv_sharing else None
                ),
                use_cache=use_cache,
            )

        # Final processing
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute router loss for training
        router_loss = self._compute_router_loss(router_logits, recursion_depths)

        if return_dict:
            return {
                "logits": logits,
                "recursion_depths": recursion_depths,
                "router_loss": router_loss,
                "router_logits": router_logits,
            }
        else:
            return logits

    def _create_attention_mask(
        self, base_mask: torch.Tensor, depth_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create attention mask that considers both padding and recursion depth."""
        # Expand masks for attention computation
        batch_size, seq_len = base_mask.shape

        # Base causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=base_mask.device))

        # Combine with padding mask
        attention_mask = base_mask.unsqueeze(1) * base_mask.unsqueeze(2) * causal_mask

        # Apply depth mask (tokens can only attend to active tokens at this depth)
        depth_attention = depth_mask.unsqueeze(1) * depth_mask.unsqueeze(2)
        attention_mask = attention_mask * depth_attention

        # Expand for multi-head attention: [B, H, L, L]
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, self.config.num_attention_heads, seq_len, seq_len
        )

        return attention_mask

    def _compute_router_loss(
        self, router_logits: torch.Tensor, recursion_depths: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary loss for router training."""
        # Encourage diversity in recursion depth assignment
        depth_probs = F.softmax(router_logits, dim=-1)  # [B, L, max_depth]

        # Load balancing loss: encourage uniform distribution across depths
        mean_probs = depth_probs.mean(dim=(0, 1))  # [max_depth]
        uniform_target = torch.ones_like(mean_probs) / len(mean_probs)
        load_balance_loss = F.kl_div(
            mean_probs.log(), uniform_target, reduction="batchmean"
        )

        # Sparsity loss: encourage using fewer recursion steps when possible
        avg_depth = recursion_depths.float().mean()
        sparsity_loss = avg_depth / self.config.max_recursion_depth

        return 0.01 * load_balance_loss + 0.001 * sparsity_loss


# Example usage and testing
if __name__ == "__main__":
    # Create model configuration
    config = MoRConfig(
        vocab_size=1000,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=6,
        max_recursion_depth=3,
    )

    # Initialize model
    model = MixtureOfRecursions(config)

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Recursion depths: {outputs['recursion_depths']}")
    print(f"Router loss: {outputs['router_loss'].item():.4f}")
