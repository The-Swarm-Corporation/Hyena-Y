"""
Hyena-Y: A convolution-based multi-hybrid architecture optimized for edge devices.

Hyena-Y is a variant of the Hyena family that excludes convolutions in the feature
groups (gates) while preserving the inner convolution. Combined with GQA Transformer 
blocks (1/3 of the total layers), it provides superior efficiency-quality trade-offs
for edge deployment.

This implementation is based on the research paper:
"Convolutional Multi-Hybrids for Edge Devices" by Liquid Science team.

Author: Claude 3.7 Sonnet
Date: 2025-04-24
"""

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger
from einops import repeat


class HyenaYConvolution(nn.Module):
    """
    Hyena-Y Convolution Module.
    
    This implements the Hyena-Y variant which excludes convolutions in the feature 
    groups (gates) while preserving the inner convolution.
    
    Args:
        dim (int): Hidden dimension
        short_filter_length (int): Length of the learned short explicit convolution filters
        max_seq_len (int): Maximum sequence length
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal convolution. Defaults to True.
    """
    
    def __init__(
        self,
        dim: int,
        short_filter_length: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        
        logger.info(f"Initializing HyenaYConvolution with dim={dim}, short_filter_length={short_filter_length}")
        
        self.dim = dim
        self.causal = causal
        self.short_filter_length = short_filter_length
        self.max_seq_len = max_seq_len
        
        # Projections for input
        self.projection_u = nn.Linear(dim, dim)
        self.projection_v = nn.Linear(dim, dim)
        
        # Learned short explicit (SE) filter - one per feature dimension
        self.filter = nn.Parameter(torch.randn(dim, 1, short_filter_length))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Hyena-Y convolution.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        batch, seq_len, dim = x.shape
        
        # Apply projections
        u = self.projection_u(x)  # [batch, seq_len, dim]
        v = self.projection_v(x)  # [batch, seq_len, dim]
        
        # Apply inner convolution
        # Each feature dimension processed separately
        u_reshaped = u.permute(0, 2, 1)  # [batch, dim, seq_len]
        
        # Ensure filter doesn't exceed sequence length
        effective_filter = self.filter
        if self.short_filter_length > seq_len:
            effective_filter = effective_filter[:, :, :seq_len]
        
        # Apply 1D convolution with padding for causal convolution
        padding_size = self.short_filter_length - 1 if self.causal else (self.short_filter_length - 1) // 2
        u_padded = F.pad(u_reshaped, (padding_size, 0) if self.causal else (padding_size, padding_size))
        
        # Process each feature dimension with its own filter
        u_filtered = F.conv1d(u_padded, effective_filter, groups=dim)
        
        # Convert back to original shape
        u_filtered = u_filtered.permute(0, 2, 1)  # [batch, seq_len, dim]
        
        # Element-wise multiplication (no convolution in the gates)
        y = u_filtered * v
        
        return self.dropout(y)


class HyenaYBlock(nn.Module):
    """
    Hyena-Y Block combining layer normalization and Hyena-Y convolution.
    
    Args:
        dim (int): Hidden dimension
        short_filter_length (int): Length of the learned short explicit convolution filters
        max_seq_len (int): Maximum sequence length
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal convolution. Defaults to True.
    """
    
    def __init__(
        self,
        dim: int,
        short_filter_length: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        
        logger.info(f"Initializing HyenaYBlock with dim={dim}")
        
        self.norm = nn.LayerNorm(dim)
        self.hyena_y_conv = HyenaYConvolution(
            dim=dim,
            short_filter_length=short_filter_length,
            max_seq_len=max_seq_len,
            dropout=dropout,
            causal=causal,
        )
        
        # Residual projection (can be set to identity if dimensions match)
        self.residual_proj = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Hyena-Y block.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        residual = x
        x = self.norm(x)
        x = self.hyena_y_conv(x)
        return self.residual_proj(residual) + x


class GQAAttention(nn.Module):
    """
    Grouped Query Attention module.
    
    Implements the GQA mechanism as described in "GQA: Training Generalized Multi-Query 
    Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023).
    
    Args:
        dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        num_kv_heads (int): Number of key-value heads (smaller than num_heads)
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        logger.info(f"Initializing GQAAttention with dim={dim}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_groups = num_heads // num_kv_heads
        self.causal = causal
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(dim, self.head_dim * num_kv_heads)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for GQA attention.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape q, k, v for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand k and v to match q shape
        if self.kv_groups > 1:
            k = repeat(k, 'b h s d -> b (h g) s d', g=self.kv_groups)
            v = repeat(v, 'b h s d -> b (h g) s d', g=self.kv_groups)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        
        return out


class GQATransformerBlock(nn.Module):
    """
    Transformer block with Grouped Query Attention.
    
    Args:
        dim (int): Hidden dimension
        num_heads (int): Number of attention heads
        num_kv_heads (int): Number of key-value heads
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        causal=config.causal,
    )
    
    return model, config (int): Feed-forward network dimension
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal attention. Defaults to True.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        
        logger.info(f"Initializing GQATransformerBlock with dim={dim}, ffn_dim={ffn_dim}")
        
        # First normalization and attention
        self.norm1 = nn.LayerNorm(dim)
        self.attention = GQAAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            causal=causal,
        )
        
        # Second normalization and feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for GQA Transformer block.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        # First sub-block: attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = residual + x
        
        # Second sub-block: feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class HyenaY(nn.Module):
    """
    Hyena-Y model: A convolution-based multi-hybrid architecture optimized for edge devices.
    
    This architecture combines Hyena-Y blocks (which exclude convolutions in the feature groups/gates)
    with GQA Transformer blocks in a 2:1 ratio (2/3 Hyena-Y, 1/3 GQA).
    
    Args:
        vocab_size (int): Size of the vocabulary
        dim (int): Hidden dimension
        depth (int): Total number of blocks
        short_filter_length (int): Length of the learned short explicit convolution filters
        max_seq_len (int): Maximum sequence length
        num_heads (int): Number of attention heads in GQA blocks
        num_kv_heads (int): Number of key-value heads in GQA blocks
        ffn_dim (int): Feed-forward network dimension in GQA blocks
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal convolution and attention. Defaults to True.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        short_filter_length: int,
        max_seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        
        logger.info(f"Initializing HyenaY model with vocab_size={vocab_size}, dim={dim}, depth={depth}")
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # Calculate how many of each block type we need
        num_hyena_blocks = 2 * depth // 3  # 2/3 of the total blocks
        num_transformer_blocks = depth - num_hyena_blocks  # 1/3 of the total blocks
        
        logger.info(f"Creating {num_hyena_blocks} Hyena-Y blocks and {num_transformer_blocks} GQA Transformer blocks")
        
        # Build the layers
        self.layers = nn.ModuleList()
        
        # Alternate between Hyena-Y and GQA blocks, with twice as many Hyena-Y blocks
        hyena_counter = 0
        transformer_counter = 0
        
        for i in range(depth):
            # Decide which type of block to add
            # Add two Hyena-Y blocks, then one GQA block
            if i % 3 < 2 and hyena_counter < num_hyena_blocks:
                self.layers.append(
                    HyenaYBlock(
                        dim=dim,
                        short_filter_length=short_filter_length,
                        max_seq_len=max_seq_len,
                        dropout=dropout,
                        causal=causal,
                    )
                )
                hyena_counter += 1
            else:
                self.layers.append(
                    GQATransformerBlock(
                        dim=dim,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                        causal=causal,
                    )
                )
                transformer_counter += 1
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between token embedding and LM head
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights of the model.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        inputs: Tensor,
        positions: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for Hyena-Y model.
        
        Args:
            inputs (Tensor): Either input token indices of shape [batch_size, seq_len]
                           or pre-embedded inputs of shape [batch_size, seq_len, dim]
            positions (Optional[Tensor], optional): Custom position indices. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask for GQA blocks. Defaults to None.
            
        Returns:
            Tensor: Logits for next token prediction of shape [batch_size, seq_len, vocab_size]
        """
        # Check if inputs are token IDs or embeddings
        if inputs.dim() == 2:
            # Inputs are token IDs [batch_size, seq_len]
            batch_size, seq_len = inputs.shape
            
            # Get token embeddings
            x = self.token_embedding(inputs)
            
            # Add position embeddings
            if positions is None:
                positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
            position_embs = self.position_embedding(positions)
            x = x + position_embs
        else:
            # Inputs are already embeddings [batch_size, seq_len, dim]
            x = inputs
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GQATransformerBlock):
                x = layer(x, attention_mask)
            else:
                x = layer(x)
        
        # Final norm and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Load pretrained weights from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle potential key mismatches between checkpoint and model
        model_state_dict = self.state_dict()
        pretrained_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Filter out unexpected keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        
        # Load weights
        self.load_state_dict(pretrained_state_dict, strict=False)
        
        # Log which keys were not found
        missing_keys = set(model_state_dict.keys()) - set(pretrained_state_dict.keys())
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate text from the model.
        
        Args:
            input_ids (Tensor): Input token indices of shape [batch_size, seq_len]
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Number of highest probability tokens to keep. Defaults to 50.
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.9.
            do_sample (bool, optional): Whether to sample or take the most likely token. Defaults to True.
            eos_token_id (Optional[int], optional): End-of-sequence token ID. Defaults to None.
            
        Returns:
            Tensor: Generated token indices of shape [batch_size, seq_len + max_new_tokens]
        """
        logger.info(f"Generating {max_new_tokens} tokens with temperature={temperature}, top_k={top_k}, top_p={top_p}")
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated = input_ids.clone()
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get the current length
            cur_len = generated.shape[1]
            
            # Truncate if exceeding maximum sequence length
            if cur_len >= self.max_seq_len:
                inputs = generated[:, -self.max_seq_len:]
                positions = torch.arange(self.max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                inputs = generated
                positions = torch.arange(cur_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Forward pass to get logits
            logits = self.forward(inputs, positions)
            
            # Take the logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits.masked_fill_(indices_to_remove, float('-inf'))
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Back to unsorted indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits.masked_fill_(indices_to_remove, float('-inf'))
            
            # Sample or greedy selection
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the new token to the sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated


class HyenaYConfig:
    """
    Configuration class for Hyena-Y model.
    
    Args:
        vocab_size (int): Size of the vocabulary
        dim (int): Hidden dimension
        depth (int): Total number of blocks
        short_filter_length (int): Length of the learned short explicit convolution filters
        max_seq_len (int): Maximum sequence length
        num_heads (int): Number of attention heads in GQA blocks
        num_kv_heads (int): Number of key-value heads in GQA blocks
        ffn_dim (int): Feed-forward network dimension in GQA blocks
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal convolution and attention. Defaults to True.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 768,
        depth: int = 24,
        short_filter_length: int = 64,
        max_seq_len: int = 2048,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.short_filter_length = short_filter_length
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.causal = causal
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HyenaYConfig":
        """
        Create a config from a dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters
            
        Returns:
            HyenaYConfig: Configuration object
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary containing configuration parameters
        """
        return self.__dict__.copy()
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the config to a directory.
        
        Args:
            save_directory (str): Directory where to save the config
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "HyenaYConfig":
        """
        Load a config from a pretrained model directory.
        
        Args:
            pretrained_model_name_or_path (str): Directory or model name
            
        Returns:
            HyenaYConfig: Configuration object
        """
        import os
        import json
        
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


def create_hyena_y_7b() -> Tuple[HyenaY, HyenaYConfig]:
    """
    Create a 7B parameter Hyena-Y model.
    
    Returns:
        Tuple[HyenaY, HyenaYConfig]: Model and configuration
    """
    logger.info("Creating Hyena-Y 7B model")
    
    config = HyenaYConfig(
        vocab_size=32000,
        dim=4096,
        depth=32,
        short_filter_length=64,
        max_seq_len=4096,
        num_heads=32,
        num_kv_heads=8,
        ffn_dim=11008,
        dropout=0.0,
        causal=True,
    )
    
    model = HyenaY(
        vocab_size=config.vocab_size,
        dim=config.dim,
        depth=config.depth,
        short_filter_length=config.short_filter_length,
        max_seq_len=config.max_seq_len,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        causal=config.causal,
    )
    
    return model, config


def create_hyena_y_1b() -> Tuple[HyenaY, HyenaYConfig]:
    """
    Create a 1B parameter Hyena-Y model for edge devices.
    
    Returns:
        Tuple[HyenaY, HyenaYConfig]: Model and configuration
    """
    logger.info("Creating Hyena-Y 1B model for edge devices")
    
    config = HyenaYConfig(
        vocab_size=32000,
        dim=2048,
        depth=24,
        short_filter_length=32,
        max_seq_len=2048,
        num_heads=16,
        num_kv_heads=4,
        ffn_dim=5632,
        dropout=0.0,
        causal=True,
    )
    
    model = HyenaY(
        vocab_size=config.vocab_size,
        dim=config.dim,
        depth=config.depth,
        short_filter_length=config.short_filter_length,
        max_seq_len=config.max_seq_len,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        causal=config.causal,
    )
    
    return model, config

model, config = create_hyena_y_1b()

x = torch.randn(1, 1024, 2048)

y = model(x)

print(y.shape)
