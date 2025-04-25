"""
HyenaY-Vision: A Visual-Language Embedding Model using Hyena-Y Backbone

This module implements a visual-language embedding model with the Hyena-Y architecture
as its backbone. It supports flexible image sizes and resolutions and can be used for
multimodal tasks such as image-text matching, visual question answering, and more.

Author: Claude 3.7 Sonnet
Date: 2025-04-25
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
import math
from loguru import logger

# Import the Hyena-Y model from the base implementation
from hyena_y import HyenaY, HyenaYConfig, HyenaYBlock


class PatchEmbedding(nn.Module):
    """
    Flexible patch embedding module for variable input image sizes.
    
    This module converts a batch of images into a sequence of patch embeddings.
    It supports variable image sizes by dynamically computing the number of patches.
    
    Args:
        img_size (int, optional): Default image size. Defaults to 224.
        patch_size (int): Size of patches to extract. Defaults to 16.
        in_channels (int): Number of input image channels. Defaults to 3 (RGB).
        embed_dim (int): Embedding dimension. Defaults to 768.
        adaptive_pooling (bool): Whether to use adaptive pooling for different resolutions. Defaults to True.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        adaptive_pooling: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.adaptive_pooling = adaptive_pooling
        
        # Compute default grid size (will be overridden for non-standard image sizes)
        self.default_grid_size = img_size // patch_size
        self.default_num_patches = self.default_grid_size ** 2
        
        # Convolutional layer for patch extraction
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Adaptive pooling for handling arbitrary image sizes
        if adaptive_pooling:
            self.pool = nn.AdaptiveAvgPool2d((None, None))
        
        # Layer normalization for patch embeddings
        self.norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Initialized PatchEmbedding with patch_size={patch_size}, embed_dim={embed_dim}")
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Forward pass for the patch embedding module.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tuple[Tensor, Tuple[int, int]]: 
                - Embedded patches of shape [batch_size, num_patches, embed_dim]
                - Grid size as (height_in_patches, width_in_patches)
        """
        B, C, H, W = x.shape
        
        # Check input channels
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        
        # Compute grid size for the input image
        height_in_patches = H // self.patch_size
        width_in_patches = W // self.patch_size
        
        # Apply adaptive pooling if needed
        if self.adaptive_pooling and (H % self.patch_size != 0 or W % self.patch_size != 0):
            # Adjust dimensions to be divisible by patch_size
            target_H = math.ceil(H / self.patch_size) * self.patch_size
            target_W = math.ceil(W / self.patch_size) * self.patch_size
            
            # Interpolate image to make dimensions divisible by patch_size
            x = F.interpolate(
                x, size=(target_H, target_W), 
                mode='bilinear', align_corners=False
            )
            
            # Update grid size
            height_in_patches = target_H // self.patch_size
            width_in_patches = target_W // self.patch_size
        
        # Project patches
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        
        # Reshape to sequence of patches
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply normalization
        x = self.norm(x)
        
        return x, (height_in_patches, width_in_patches)


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for image patches.

    This module creates learnable 2D positional encodings for image patches,
    supporting variable grid sizes.
    
    Args:
        max_h (int): Maximum height in patches. Defaults to 14.
        max_w (int): Maximum width in patches. Defaults to 14.
        embed_dim (int): Embedding dimension. Defaults to 768.
    """
    
    def __init__(
        self,
        max_h: int = 14,
        max_w: int = 14,
        embed_dim: int = 768,
    ):
        super().__init__()
        
        self.max_h = max_h
        self.max_w = max_w
        self.embed_dim = embed_dim
        
        # Create learnable positional embeddings
        self.pos_embed_h = nn.Parameter(torch.zeros(1, max_h, embed_dim // 2))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, max_w, embed_dim // 2))
        
        # Initialize with sinusoidal encoding
        self._init_weights()
        
        logger.info(f"Initialized 2D positional encoding with max_h={max_h}, max_w={max_w}")
    
    def _init_weights(self) -> None:
        """Initialize positional embeddings with sinusoidal encoding"""
        h_positions = torch.arange(0, self.max_h).float()
        w_positions = torch.arange(0, self.max_w).float()
        
        div_term = torch.exp(
            torch.arange(0, self.embed_dim // 2, 2).float() * (-math.log(10000.0) / (self.embed_dim // 2))
        )
        
        # Initialize height embeddings
        pos_h = torch.zeros(1, self.max_h, self.embed_dim // 2)
        for i in range(0, self.embed_dim // 4, 2):
            pos_h[0, :, i] = torch.sin(h_positions * div_term[i // 2])
            pos_h[0, :, i + 1] = torch.cos(h_positions * div_term[i // 2])
        
        # Initialize width embeddings
        pos_w = torch.zeros(1, self.max_w, self.embed_dim // 2)
        for i in range(0, self.embed_dim // 4, 2):
            pos_w[0, :, i] = torch.sin(w_positions * div_term[i // 2])
            pos_w[0, :, i + 1] = torch.cos(w_positions * div_term[i // 2])
        
        self.pos_embed_h.data.copy_(pos_h)
        self.pos_embed_w.data.copy_(pos_w)
    
    def forward(self, grid_size: Tuple[int, int]) -> Tensor:
        """
        Generate positional encodings for a specific grid size.
        
        Args:
            grid_size (Tuple[int, int]): Grid size as (height_in_patches, width_in_patches)
            
        Returns:
            Tensor: Positional encodings of shape [1, height_in_patches * width_in_patches, embed_dim]
        """
        h, w = grid_size
        
        # Limit to maximum dimensions
        h = min(h, self.max_h)
        w = min(w, self.max_w)
        
        # Get positional embeddings for h and w
        pos_h = self.pos_embed_h[:, :h, :]  # [1, h, embed_dim//2]
        pos_w = self.pos_embed_w[:, :w, :]  # [1, w, embed_dim//2]
        
        # Create 2D positional encoding
        pos_h = pos_h.unsqueeze(2).expand(-1, -1, w, -1)  # [1, h, w, embed_dim//2]
        pos_w = pos_w.unsqueeze(1).expand(-1, h, -1, -1)  # [1, h, w, embed_dim//2]
        
        # Combine h and w positional encodings
        pos = torch.cat([pos_h, pos_w], dim=-1)  # [1, h, w, embed_dim]
        
        # Reshape to sequence
        pos = pos.reshape(1, h * w, self.embed_dim)
        
        return pos


class VisualHyenaYBlock(nn.Module):
    """
    Adapts the Hyena-Y block for visual processing.
    
    This block extends the HyenaYBlock to include positional encodings
    specific to visual data.
    
    Args:
        dim (int): Hidden dimension
        short_filter_length (int): Length of the learned short explicit convolution filters
        max_seq_len (int): Maximum sequence length
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        causal (bool, optional): Whether to use causal convolution. Defaults to False for images.
    """
    
    def __init__(
        self,
        dim: int,
        short_filter_length: int,
        max_seq_len: int,
        dropout: float = 0.0,
        causal: bool = False,  # Non-causal by default for images
    ):
        super().__init__()
        
        logger.info(f"Initializing VisualHyenaYBlock with dim={dim}")
        
        # Use the original HyenaYBlock
        self.hyena_y_block = HyenaYBlock(
            dim=dim,
            short_filter_length=short_filter_length,
            max_seq_len=max_seq_len,
            dropout=dropout,
            causal=causal,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for Visual Hyena-Y block.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, dim]
        """
        return self.hyena_y_block(x)


class VisionEncoder(nn.Module):
    """
    Vision encoder using Hyena-Y blocks for image encoding.
    
    This module encodes an image into a sequence of embeddings.
    
    Args:
        img_size (int, optional): Default image size. Defaults to 224.
        patch_size (int): Size of patches to extract. Defaults to 16.
        in_channels (int): Number of input image channels. Defaults to 3 (RGB).
        embed_dim (int): Embedding dimension. Defaults to 768.
        depth (int): Number of Hyena-Y blocks. Defaults to 12.
        short_filter_length (int): Length of the learned short explicit convolution filters. Defaults to 32.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        adaptive_pooling (bool): Whether to use adaptive pooling for different resolutions. Defaults to True.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        short_filter_length: int = 32,
        dropout: float = 0.0,
        adaptive_pooling: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            adaptive_pooling=adaptive_pooling,
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(
            max_h=img_size // patch_size,
            max_w=img_size // patch_size,
            embed_dim=embed_dim,
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Hyena-Y blocks for vision
        max_patches = (img_size // patch_size) ** 2 + 1  # +1 for cls token
        self.blocks = nn.ModuleList([
            VisualHyenaYBlock(
                dim=embed_dim,
                short_filter_length=short_filter_length,
                max_seq_len=max_patches,
                dropout=dropout,
                causal=False,  # Non-causal for images
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize parameters
        self._init_weights()
        
        logger.info(f"Initialized VisionEncoder with depth={depth}, embed_dim={embed_dim}")
    
    def _init_weights(self) -> None:
        """Initialize the weights"""
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the vision encoder.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor: Encoded image embeddings of shape [batch_size, num_patches + 1, embed_dim]
                   where the first token is the CLS token
        """
        # Get patch embeddings
        patches, grid_size = self.patch_embed(x)  # B, num_patches, embed_dim
        
        # Get positional encodings
        pos = self.pos_encoding(grid_size)  # 1, num_patches, embed_dim
        
        # Add positional encodings
        patches = patches + pos
        
        # Add class token
        batch_size = patches.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, patches], dim=1)  # B, num_patches + 1, embed_dim
        
        # Apply Hyena-Y blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x


class ImageTextEmbedModel(nn.Module):
    """
    Combined image-text embedding model using Hyena-Y as backbone.
    
    This model encodes both images and text into a shared embedding space
    for multimodal tasks.
    
    Args:
        vision_config (Dict): Configuration for the vision encoder
        text_config (HyenaYConfig): Configuration for the text encoder
    """
    
    def __init__(
        self,
        vision_config: Dict,
        text_config: HyenaYConfig,
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(**vision_config)
        
        # Text encoder (Hyena-Y model)
        self.text_encoder = HyenaY(
            vocab_size=text_config.vocab_size,
            dim=text_config.dim,
            depth=text_config.depth,
            short_filter_length=text_config.short_filter_length,
            max_seq_len=text_config.max_seq_len,
            num_heads=text_config.num_heads,
            num_kv_heads=text_config.num_kv_heads,
            ffn_dim=text_config.ffn_dim,
            dropout=text_config.dropout,
            causal=text_config.causal,
        )
        
        # Projection layers for alignment
        self.vision_proj = nn.Linear(vision_config["embed_dim"], text_config.dim)
        self.text_proj = nn.Linear(text_config.dim, text_config.dim)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        logger.info("Initialized ImageTextEmbedModel")
    
    def encode_image(self, images: Tensor) -> Tensor:
        """
        Encode images.
        
        Args:
            images (Tensor): Input images of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor: Image embeddings of shape [batch_size, embed_dim]
        """
        # Encode images with vision encoder
        vision_embeddings = self.vision_encoder(images)  # B, num_patches + 1, embed_dim
        
        # Use CLS token for image representation
        cls_embedding = vision_embeddings[:, 0]  # B, embed_dim
        
        # Project to text dimension
        image_features = self.vision_proj(cls_embedding)  # B, text_dim
        
        # Normalize
        image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features
    
    def encode_text(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode text.
        
        Args:
            input_ids (Tensor): Input token IDs of shape [batch_size, seq_len]
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Tensor: Text embeddings of shape [batch_size, embed_dim]
        """
        # Encode text with text encoder
        text_outputs = self.text_encoder(input_ids)  # B, seq_len, vocab_size
        
        # Pool text representation (use the last token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with mask
            mask = attention_mask.unsqueeze(-1)
            text_embedding = torch.sum(text_outputs * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            # Use last token
            text_embedding = text_outputs[:, -1]
        
        # Project to aligned dimension
        text_features = self.text_proj(text_embedding)  # B, text_dim
        
        # Normalize
        text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features
    
    def forward(
        self,
        images: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for the image-text model.
        
        Args:
            images (Optional[Tensor], optional): Input images. Defaults to None.
            input_ids (Optional[Tensor], optional): Input token IDs. Defaults to None.
            attention_mask (Optional[Tensor], optional): Attention mask. Defaults to None.
            
        Returns:
            Dict[str, Tensor]: Output dictionary with image_features, text_features,
                               and optionally logits_per_image and logits_per_text
        """
        output_dict = {}
        
        # Encode images if provided
        if images is not None:
            image_features = self.encode_image(images)
            output_dict["image_features"] = image_features
        
        # Encode text if provided
        if input_ids is not None:
            text_features = self.encode_text(input_ids, attention_mask)
            output_dict["text_features"] = text_features
        
        # Compute logits if both modalities are provided
        if images is not None and input_ids is not None:
            # Compute similarity scores
            logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
            logits_per_text = logits_per_image.t()
            
            output_dict["logits_per_image"] = logits_per_image
            output_dict["logits_per_text"] = logits_per_text
        
        return output_dict
    
    def compute_contrastive_loss(self, logits_per_image: Tensor) -> Tensor:
        """
        Compute contrastive loss.
        
        Args:
            logits_per_image (Tensor): Similarity scores of shape [batch_size, batch_size]
            
        Returns:
            Tensor: Contrastive loss
        """
        batch_size = logits_per_image.shape[0]
        targets = torch.arange(batch_size, device=logits_per_image.device)
        
        # Symmetric loss
        loss_i = F.cross_entropy(logits_per_image, targets)
        loss_t = F.cross_entropy(logits_per_image.t(), targets)
        
        return (loss_i + loss_t) / 2.0
    
    def generate_image_captions(
        self,
        images: Tensor,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate image captions.
        
        Args:
            images (Tensor): Input images of shape [batch_size, channels, height, width]
            input_ids (Tensor): Initial token IDs of shape [batch_size, seq_len]
            max_new_tokens (int, optional): Maximum number of new tokens. Defaults to 20.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k filtering. Defaults to 50.
            top_p (float, optional): Top-p (nucleus) filtering. Defaults to 0.9.
            do_sample (bool, optional): Whether to sample or greedy search. Defaults to True.
            eos_token_id (Optional[int], optional): End-of-sequence token ID. Defaults to None.
            
        Returns:
            Tensor: Generated token IDs of shape [batch_size, seq_len + max_new_tokens]
        """
        # Encode images
        image_features = self.encode_image(images)  # B, text_dim
        
        # Prepare for text generation
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize sequence with input_ids
        generated = input_ids.clone()
        
        # Get the embeddings for the input tokens
        text_embeddings = self.text_encoder.token_embedding(input_ids)
        
        # Add image features to the first token embedding
        text_embeddings[:, 0] = text_embeddings[:, 0] + image_features
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get the current length
            cur_len = generated.shape[1]
            
            # Truncate if exceeding maximum sequence length
            if cur_len >= self.text_encoder.max_seq_len:
                inputs = generated[:, -self.text_encoder.max_seq_len:]
                positions = torch.arange(self.text_encoder.max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                inputs = generated
                positions = torch.arange(cur_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Forward pass with preprocessed embeddings for the first batch item
            if cur_len <= seq_len:
                # Use preprocessed embeddings for initial tokens
                logits = self.text_encoder(text_embeddings, positions)
            else:
                # Regular forward pass for generated tokens
                logits = self.text_encoder(inputs, positions)
            
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


class HyenaYVisionConfig:
    """
    Configuration class for HyenaY-Vision model.
    
    Args:
        vision_config (Dict): Configuration for the vision encoder
        text_config (HyenaYConfig): Configuration for the text encoder
    """
    
    def __init__(
        self,
        vision_config: Dict = None,
        text_config: Dict = None,
    ):
        # Default vision configuration
        self.vision_config = {
            "img_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "embed_dim": 768,
            "depth": 12,
            "short_filter_length": 32,
            "dropout": 0.0,
            "adaptive_pooling": True,
        }
        
        # Override with provided config
        if vision_config:
            self.vision_config.update(vision_config)
        
        # Default text configuration
        self.text_config = HyenaYConfig(
            vocab_size=32000,
            dim=768,
            depth=12,
            short_filter_length=32,
            max_seq_len=77,  # For CLIP-like functionality
            num_heads=12,
            num_kv_heads=4,
            ffn_dim=3072,
            dropout=0.0,
            causal=True,
        )
        
        # Override with provided config
        if text_config:
            for key, value in text_config.items():
                setattr(self.text_config, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HyenaYVisionConfig":
        """
        Create a config from a dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters
            
        Returns:
            HyenaYVisionConfig: Configuration object
        """
        vision_config = config_dict.get("vision_config", {})
        text_config = config_dict.get("text_config", {})
        return cls(vision_config=vision_config, text_config=text_config)


def create_hyena_y_vision() -> Tuple[ImageTextEmbedModel, HyenaYVisionConfig]:
    """
    Create a HyenaY-Vision model.
    
    Returns:
        Tuple[ImageTextEmbedModel, HyenaYVisionConfig]: Model and configuration
    """
    logger.info("Creating HyenaY-Vision model")
    
    # Create configuration
    config = HyenaYVisionConfig()
    
    # Create model
    model = ImageTextEmbedModel(
        vision_config=config.vision_config,
        text_config=config.text_config,
    )
    
    return model, config


def create_hyena_y_vision_large() -> Tuple[ImageTextEmbedModel, HyenaYVisionConfig]:
    """
    Create a large HyenaY-Vision model.
    
    Returns:
        Tuple[ImageTextEmbedModel, HyenaYVisionConfig]: Model and configuration
    """
    logger.info("Creating HyenaY-Vision Large model")
    
    # Vision configuration
    vision_config = {
        "img_size": 336,
        "patch_size": 14,
        "in_channels": 3,
        "embed_dim": 1024,
        "depth": 24,
        "short_filter_length": 64,
        "dropout": 0.1,
        "adaptive_pooling": True,
    }
    
    # Text configuration
    text_config = {
        "vocab_size": 32000,
        "dim": 1024,
        "depth": 24,
        "short_filter_length": 64,
        "max_seq_len": 77,
        "num_heads": 16,
        "num_kv_heads": 4,
        "ffn_dim": 4096,
        "dropout": 0.1,
        "causal": True,
    }
    
    # Create configuration
    config = HyenaYVisionConfig(
        vision_config=vision_config,
        text_config=text_config,
    )
    
    # Create model
    model = ImageTextEmbedModel(
        vision_config=config.vision_config,
        text_config=config.text_config,
    )
    
    return model, config


# Example usage

def process_image(image_path: str, device: str = "cuda") -> Tensor:
    """
    Process an image for the model.
    
    Args:
        image_path (str): Path to the image
        device (str, optional): Device to process on. Defaults to "cuda".
        
    Returns:
        Tensor: Processed image tensor
    """
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor


def process_text(text: str, tokenizer) -> Tuple[Tensor, Tensor]:
    """
    Process text for the model.
    
    Args:
        text (str): Input text
        tokenizer: Tokenizer for text processing
        
    Returns:
        Tuple[Tensor, Tensor]: Input IDs and attention mask
    """
    # Tokenize text
    encoding = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )
    
    return encoding["input_ids"], encoding["attention_mask"]


def demo_image_text_matching():
    """
    Demonstrate image-text matching with HyenaY-Vision.
    """
    import torch
    from transformers import AutoTokenizer
    
    # Create model
    model, config = create_hyena_y_vision()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Example image and text
    image_path = "example_image.jpg"
    texts = [
        "A dog running in a field",
        "A cat sleeping on a couch",
        "A mountain landscape with snow"
    ]
    
    # Process image
    image = process_image(image_path, device)
    
    # Process texts
    input_ids_list = []
    attention_mask_list = []
    
    for text in texts:
        input_ids, attention_mask = process_text(text, tokenizer)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    
    input_ids = torch.cat(input_ids_list, dim=0).to(device)
    attention_mask = torch.cat(attention_mask_list, dim=0).to(device)
    
    # Compute image-text matching scores
    with torch.no_grad():
        outputs = model(
            images=image.expand(len(texts), -1, -1, -1),
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Get similarity scores
    logits_per_image = outputs["logits_per_image"]
    probs = torch.softmax(logits_per_image, dim=-1)
    
    # Print results
    print("Image-text matching probabilities:")
    for i, text in enumerate(texts):
        print(f"{text}: {probs[0, i].item():.4f}")


def demo_image_captioning():
    """
    Demonstrate image captioning with HyenaY-Vision.
    """
    import torch
    from transformers import AutoTokenizer
    
    # Create model
    model, config = create_hyena_y_vision()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Example image
    image_path = "example_image.jpg"
    
    # Process image
    image = process_image(image_path, device)
    
    # Prepare input prompt
    prompt = "This image shows"
    input_ids, _ = process_text(prompt, tokenizer)
    input_ids = input_ids.to(device)
    
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate_image_captions(
            images=image,
            input_ids=input_ids,
            max_new_tokens=30,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Print result
    print(f"Generated caption: {caption}")


class HyenaYVisionProcessor:
    """
    Processor for HyenaY-Vision model.
    
    This class handles image and text preprocessing for the model.
    
    Args:
        image_size (int, optional): Image size. Defaults to 224.
        tokenizer_name (str, optional): Name of the tokenizer. Defaults to "openai/clip-vit-base-patch32".
        max_length (int, optional): Maximum text length. Defaults to 77.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ):
        from transformers import AutoTokenizer
        import torchvision.transforms as transforms
        
        self.image_size = image_size
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Define image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def preprocess_image(self, image_path: str, device: str = "cuda") -> Tensor:
        """
        Preprocess image for the model.
        
        Args:
            image_path (str): Path to the image
            device (str, optional): Device to process on. Defaults to "cuda".
            
        Returns:
            Tensor: Processed image tensor
        """
        from PIL import Image
        
        # Handle different input types
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise ValueError("Unsupported image input type")
        
        # Transform image
        image_tensor = self.image_transform(image).unsqueeze(0).to(device)
        
        return image_tensor
    
    def preprocess_text(self, text: str, device: str = "cuda") -> Tuple[Tensor, Tensor]:
        """
        Preprocess text for the model.
        
        Args:
            text (str): Input text
            device (str, optional): Device to process on. Defaults to "cuda".
            
        Returns:
            Tuple[Tensor, Tensor]: Input IDs and attention mask
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        return input_ids, attention_mask
    
    def batch_process_text(self, texts: List[str], device: str = "cuda") -> Tuple[Tensor, Tensor]:
        """
        Process a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            device (str, optional): Device to process on. Defaults to "cuda".
            
        Returns:
            Tuple[Tensor, Tensor]: Batched input IDs and attention mask
        """
        # Tokenize batch of texts
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        return input_ids, attention_mask
    
    def decode_ids(self, token_ids: Tensor) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids (Tensor): Token IDs
            
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class HyenaYVisionImageSearch:
    """
    Image search using HyenaY-Vision embeddings.
    
    This class enables efficient image retrieval based on text queries
    or image similarity.
    
    Args:
        model (ImageTextEmbedModel): HyenaY-Vision model
        processor (HyenaYVisionProcessor): HyenaY-Vision processor
        device (str, optional): Device to process on. Defaults to "cuda".
    """
    
    def __init__(
        self,
        model: ImageTextEmbedModel,
        processor: HyenaYVisionProcessor,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Storage for image embeddings
        self.image_paths = []
        self.image_embeddings = []
    
    def index_images(self, image_paths: List[str]) -> None:
        """
        Index a list of images.
        
        Args:
            image_paths (List[str]): List of paths to images
        """
        import torch
        
        # Process and index images
        for image_path in image_paths:
            # Process image
            image = self.processor.preprocess_image(image_path, self.device)
            
            # Compute embedding
            with torch.no_grad():
                image_embedding = self.model.encode_image(image)
            
            # Store path and embedding
            self.image_paths.append(image_path)
            self.image_embeddings.append(image_embedding.cpu())
        
        # Stack embeddings for efficient search
        self.image_embeddings = torch.cat(self.image_embeddings, dim=0)
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search images by text query.
        
        Args:
            query (str): Text query
            top_k (int, optional): Number of top results to return. Defaults to 5.
            
        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) tuples
        """
        import torch
        
        # Process query
        input_ids, attention_mask = self.processor.preprocess_text(query, self.device)
        
        # Compute query embedding
        with torch.no_grad():
            text_embedding = self.model.encode_text(input_ids, attention_mask)
        
        # Compute similarity scores
        text_embedding = text_embedding.cpu()
        similarities = torch.matmul(self.image_embeddings, text_embedding.t()).squeeze()
        
        # Get top-k results
        if len(similarities.shape) == 0:
            similarities = similarities.unsqueeze(0)
        
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.image_paths)))
        
        # Return results
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.image_paths[idx], score))
        
        return results
    
    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search images by image similarity.
        
        Args:
            image_path (str): Path to query image
            top_k (int, optional): Number of top results to return. Defaults to 5.
            
        Returns:
            List[Tuple[str, float]]: List of (image_path, similarity_score) tuples
        """
        import torch
        
        # Process query image
        image = self.processor.preprocess_image(image_path, self.device)
        
        # Compute query embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
        
        # Compute similarity scores
        image_embedding = image_embedding.cpu()
        similarities = torch.matmul(self.image_embeddings, image_embedding.t()).squeeze()
        
        # Get top-k results
        if len(similarities.shape) == 0:
            similarities = similarities.unsqueeze(0)
        
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.image_paths)))
        
        # Return results
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.image_paths[idx], score))
        
        return results


# Training utilities

def contrastive_loss(logits_per_image: Tensor, logits_per_text: Tensor) -> Tensor:
    """
    Compute bidirectional contrastive loss.
    
    Args:
        logits_per_image (Tensor): Image-to-text similarity scores
        logits_per_text (Tensor): Text-to-image similarity scores
        
    Returns:
        Tensor: Contrastive loss
    """
    batch_size = logits_per_image.shape[0]
    targets = torch.arange(batch_size, device=logits_per_image.device)
    
    # Image-to-text loss
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    
    # Text-to-image loss
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    
    # Total loss
    total_loss = (loss_i2t + loss_t2i) / 2
    
    return total_loss


def train_hyena_y_vision(
    model: ImageTextEmbedModel,
    train_dataloader,
    optimizer,
    device: str = "cuda",
    epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 100,
):
    """
    Train HyenaY-Vision model.
    
    Args:
        model (ImageTextEmbedModel): Model to train
        train_dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device (str, optional): Device to train on. Defaults to "cuda".
        epochs (int, optional): Number of training epochs. Defaults to 3.
        gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 1.
        log_interval (int, optional): Logging interval. Defaults to 100.
    """
    import torch
    from tqdm import tqdm
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Get batch data
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = contrastive_loss(
                outputs["logits_per_image"],
                outputs["logits_per_text"]
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if needed
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            steps += 1
            
            # Update progress bar
            if steps % log_interval == 0:
                progress_bar.set_postfix({"loss": epoch_loss / steps})
        
        # Print epoch results
        avg_epoch_loss = epoch_loss / steps
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    # Example: Create and test model
    model, config = create_hyena_y_vision()
    processor = HyenaYVisionProcessor()
    
    print("HyenaY-Vision model created successfully!")
