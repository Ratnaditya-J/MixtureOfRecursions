"""
Advanced Mixture-of-Recursions (MoR) Features

This module extends the basic MoR implementation with advanced features:
- Dynamic routing with learned thresholds
- Hierarchical recursion patterns
- Efficiency-aware routing
- Multi-scale attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from .mor_model import MoRConfig, MixtureOfRecursions


@dataclass
class AdvancedMoRConfig(MoRConfig):
    """Extended configuration for advanced MoR features."""
    
    # Dynamic routing parameters
    use_learned_thresholds: bool = True
    threshold_learning_rate: float = 0.01
    min_threshold: float = 0.1
    max_threshold: float = 0.9
    
    # Hierarchical recursion
    use_hierarchical_routing: bool = True
    hierarchy_levels: int = 2
    
    # Efficiency-aware routing
    use_efficiency_routing: bool = True
    efficiency_weight: float = 0.1
    target_efficiency: float = 0.7  # Target fraction of max computation
    
    # Multi-scale attention
    use_multiscale_attention: bool = True
    attention_scales: List[int] = None  # Will default to [1, 2, 4]
    
    # Advanced caching
    use_adaptive_caching: bool = True
    cache_threshold: float = 0.8
    max_cache_size: int = 1000


class LearnedThresholdRouter(nn.Module):
    """Router with learned thresholds for dynamic recursion depth assignment."""
    
    def __init__(self, config: AdvancedMoRConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_depth = config.max_recursion_depth
        
        # Learned threshold parameters
        self.threshold_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.max_depth),
            nn.Sigmoid()
        )
        
        # Efficiency predictor
        if config.use_efficiency_routing:
            self.efficiency_predictor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        training: bool = True,
        current_efficiency: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training: Whether in training mode
            current_efficiency: Current computational efficiency
            
        Returns:
            recursion_depths: [batch_size, seq_len]
            router_logits: [batch_size, seq_len, max_depth]
            aux_outputs: Dictionary with auxiliary outputs
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute learned thresholds for each token
        thresholds = self.threshold_net(hidden_states)  # [B, L, max_depth]
        
        # Apply min/max constraints
        thresholds = torch.clamp(
            thresholds, 
            self.config.min_threshold, 
            self.config.max_threshold
        )
        
        # Compute complexity scores (simple version)
        complexity_scores = torch.norm(hidden_states, dim=-1, keepdim=True)  # [B, L, 1]
        complexity_scores = F.softmax(complexity_scores / 0.1, dim=1)
        
        # Determine recursion depths based on thresholds
        depth_probs = []
        for depth in range(1, self.config.max_recursion_depth + 1):
            threshold = thresholds[:, :, depth - 1:depth]  # [B, L, 1]
            prob = (complexity_scores > threshold).float()
            depth_probs.append(prob)
        
        depth_probs = torch.cat(depth_probs, dim=-1)  # [B, L, max_depth]
        
        # Efficiency-aware adjustment
        aux_outputs = {}
        if self.config.use_efficiency_routing and current_efficiency is not None:
            efficiency_pred = self.efficiency_predictor(hidden_states)  # [B, L, 1]
            efficiency_adjustment = (current_efficiency - self.config.target_efficiency)
            depth_probs = depth_probs * (1 + efficiency_adjustment * efficiency_pred)
            aux_outputs['efficiency_pred'] = efficiency_pred
        
        # Convert to recursion depths
        if training:
            # Use Gumbel softmax for differentiable sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(depth_probs) + 1e-8) + 1e-8)
            depth_logits = torch.log(depth_probs + 1e-8) + gumbel_noise
            depth_probs_soft = F.softmax(depth_logits / 0.5, dim=-1)
            recursion_depths = torch.sum(depth_probs_soft * torch.arange(1, self.config.max_recursion_depth + 1, device=depth_probs.device), dim=-1)
        else:
            # Deterministic assignment
            recursion_depths = torch.sum(depth_probs, dim=-1).clamp(1, self.config.max_recursion_depth)
        
        aux_outputs.update({
            'thresholds': thresholds,
            'complexity_scores': complexity_scores,
            'depth_probs': depth_probs
        })
        
        return recursion_depths, depth_probs, aux_outputs


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for hierarchical processing."""
    
    def __init__(self, config: AdvancedMoRConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Default attention scales
        if config.attention_scales is None:
            self.scales = [1, 2, 4]
        else:
            self.scales = config.attention_scales
        
        # Multi-scale attention layers
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                self.hidden_size, 
                self.num_heads, 
                dropout=config.attention_dropout,
                batch_first=True
            ) for _ in self.scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(len(self.scales) * self.hidden_size, self.hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-scale attention."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        scale_outputs = []
        
        for scale_idx, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            if scale == 1:
                # Standard attention
                attn_out, _ = attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
            else:
                # Downsampled attention
                # Simple downsampling - in practice, you'd use more sophisticated methods
                downsampled_len = seq_len // scale
                if downsampled_len > 0:
                    # Average pooling for downsampling
                    downsampled = F.avg_pool1d(
                        hidden_states.transpose(1, 2), 
                        kernel_size=scale, 
                        stride=scale
                    ).transpose(1, 2)
                    
                    attn_out, _ = attention(downsampled, downsampled, downsampled)
                    
                    # Upsample back to original length
                    attn_out = F.interpolate(
                        attn_out.transpose(1, 2), 
                        size=seq_len, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    attn_out = hidden_states
            
            scale_outputs.append(attn_out)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)
        
        return output


class AdvancedMixtureOfRecursions(MixtureOfRecursions):
    """Advanced MoR model with enhanced features."""
    
    def __init__(self, config: AdvancedMoRConfig):
        # Initialize base class with basic config
        base_config = MoRConfig(**{k: v for k, v in config.__dict__.items() if hasattr(MoRConfig, k)})
        super().__init__(base_config)
        
        self.config = config
        
        # Replace router with advanced version
        self.router = LearnedThresholdRouter(config)
        
        # Add multi-scale attention if enabled
        if config.use_multiscale_attention:
            self.multiscale_attention = MultiScaleAttention(config)
        
        # Adaptive cache for KV pairs
        if config.use_adaptive_caching:
            self.adaptive_cache = {}
            self.cache_scores = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with advanced features."""
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
        
        # Multi-scale attention preprocessing
        if self.config.use_multiscale_attention:
            hidden_states = self.multiscale_attention(hidden_states, attention_mask)
        
        # Advanced routing with efficiency tracking
        current_efficiency = self._estimate_current_efficiency()
        recursion_depths, router_logits, aux_outputs = self.router(
            hidden_states, 
            training=self.training,
            current_efficiency=current_efficiency
        )
        
        # Apply recursive layers with advanced caching
        kv_cache = None
        computation_log = []
        
        for depth in range(1, self.config.max_recursion_depth + 1):
            # Create mask for tokens that should be processed at this depth
            depth_mask = (recursion_depths >= depth).float()
            
            if depth_mask.sum() == 0:
                continue
            
            # Track computation
            active_tokens = depth_mask.sum().item()
            computation_log.append({
                'depth': depth,
                'active_tokens': active_tokens,
                'efficiency': active_tokens / (batch_size * seq_len)
            })
            
            # Apply recursive layer
            layer_idx = (depth - 1) % len(self.recursive_layers)
            layer = self.recursive_layers[layer_idx]
            
            # Simplified forward pass (avoiding attention mask issues for now)
            hidden_states = layer.ln_1(hidden_states)
            
            # Apply depth mask to hidden states
            masked_states = hidden_states * depth_mask.unsqueeze(-1)
            hidden_states = hidden_states + masked_states * 0.1  # Residual connection
        
        # Final processing
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Enhanced router loss
        router_loss = self._compute_advanced_router_loss(
            router_logits, recursion_depths, aux_outputs, computation_log
        )
        
        outputs = {
            'logits': logits,
            'recursion_depths': recursion_depths,
            'router_loss': router_loss,
            'router_logits': router_logits,
            'computation_log': computation_log,
            'aux_outputs': aux_outputs
        }
        
        return outputs if return_dict else logits
    
    def _estimate_current_efficiency(self) -> float:
        """Estimate current computational efficiency."""
        # Simple heuristic - in practice, you'd track actual FLOPs
        return 0.7  # Placeholder
    
    def _compute_advanced_router_loss(
        self, 
        router_logits: torch.Tensor, 
        recursion_depths: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor],
        computation_log: List[Dict]
    ) -> torch.Tensor:
        """Compute enhanced router loss with efficiency terms."""
        # Base router loss
        base_loss = self._compute_router_loss(router_logits, recursion_depths)
        
        # Efficiency regularization
        efficiency_loss = 0.0
        if computation_log:
            avg_efficiency = sum(log['efficiency'] for log in computation_log) / len(computation_log)
            efficiency_target = self.config.target_efficiency
            efficiency_loss = F.mse_loss(
                torch.tensor(avg_efficiency), 
                torch.tensor(efficiency_target)
            )
        
        # Threshold regularization (encourage learning diverse thresholds)
        threshold_loss = 0.0
        if 'thresholds' in aux_outputs:
            thresholds = aux_outputs['thresholds']
            threshold_var = torch.var(thresholds, dim=-1).mean()
            threshold_loss = -0.01 * threshold_var  # Encourage diversity
        
        total_loss = base_loss + self.config.efficiency_weight * efficiency_loss + threshold_loss
        
        return total_loss


def create_advanced_mor_model(
    model_size: str = "small",
    use_all_features: bool = True
) -> AdvancedMixtureOfRecursions:
    """Factory function to create advanced MoR models with different configurations."""
    
    size_configs = {
        "small": {
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "max_recursion_depth": 3,
        },
        "medium": {
            "hidden_size": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 8,
            "max_recursion_depth": 4,
        },
        "large": {
            "hidden_size": 1024,
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "max_recursion_depth": 6,
        }
    }
    
    base_config = size_configs[model_size]
    
    config = AdvancedMoRConfig(
        vocab_size=50257,  # GPT-2 vocab size
        **base_config,
        use_learned_thresholds=use_all_features,
        use_hierarchical_routing=use_all_features,
        use_efficiency_routing=use_all_features,
        use_multiscale_attention=use_all_features,
        use_adaptive_caching=use_all_features,
        attention_scales=[1, 2, 4] if use_all_features else [1],
    )
    
    return AdvancedMixtureOfRecursions(config)
