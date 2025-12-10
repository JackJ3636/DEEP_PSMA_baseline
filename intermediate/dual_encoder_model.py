#!/usr/bin/env python3
"""
Dual-Encoder Cross-Attention U-Net for DEEP-PSMA Challenge
Handles both PSMA and FDG PET/CT with intermediate fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block with InstanceNorm and ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool + ConvBlock"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        return self.conv(torch.cat([skip, x], dim=1))


class CrossAttentionBlock(nn.Module):
    """Cross-attention mechanism for feature fusion"""
    def __init__(self, channels: int):
        super().__init__()
        reduced_channels = max(channels // 8, 16)
        self.query = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.key = nn.Conv3d(channels, reduced_channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_q, x_kv):
        B, C, D, H, W = x_q.shape
        
        # Generate query, key, value
        q = self.query(x_q).view(B, -1, D * H * W)  # (B, C', DHW)
        k = self.key(x_kv).view(B, -1, D * H * W)   # (B, C', DHW)
        v = self.value(x_kv).view(B, -1, D * H * W) # (B, C, DHW)
        
        # Compute attention
        attn = torch.bmm(q.transpose(1, 2), k)  # (B, DHW, DHW)
        attn = F.softmax(attn / (q.shape[1] ** 0.5), dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, DHW)
        out = out.view(B, C, D, H, W)
        
        # Residual connection with learnable weight
        return x_q + self.gamma * out


class DualEncoderCrossAttentionUNet(nn.Module):
    """
    Dual-encoder U-Net with cross-attention for PSMA/FDG segmentation
    Each modality has its own encoder/decoder with cross-attention fusion
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        
        # PSMA Encoder (takes 2 channels: PET + CT)
        self.psma_enc1 = ConvBlock(2, c)
        self.psma_enc2 = DownBlock(c, c * 2)
        self.psma_enc3 = DownBlock(c * 2, c * 4)
        self.psma_enc4 = DownBlock(c * 4, c * 8)
        self.psma_bottleneck = ConvBlock(c * 8, c * 16)
        
        # FDG Encoder (takes 2 channels: PET + CT)
        self.fdg_enc1 = ConvBlock(2, c)
        self.fdg_enc2 = DownBlock(c, c * 2)
        self.fdg_enc3 = DownBlock(c * 2, c * 4)
        self.fdg_enc4 = DownBlock(c * 4, c * 8)
        self.fdg_bottleneck = ConvBlock(c * 8, c * 16)
        
        # Cross-attention at multiple scales
        self.cross_attn_3 = CrossAttentionBlock(c * 4)
        self.cross_attn_4 = CrossAttentionBlock(c * 8)
        self.cross_attn_bottleneck = CrossAttentionBlock(c * 16)
        
        # PSMA Decoder
        self.psma_up4 = UpBlock(c * 16, c * 8)
        self.psma_up3 = UpBlock(c * 8, c * 4)
        self.psma_up2 = UpBlock(c * 4, c * 2)
        self.psma_up1 = UpBlock(c * 2, c)
        
        # FDG Decoder  
        self.fdg_up4 = UpBlock(c * 16, c * 8)
        self.fdg_up3 = UpBlock(c * 8, c * 4)
        self.fdg_up2 = UpBlock(c * 4, c * 2)
        self.fdg_up1 = UpBlock(c * 2, c)
        
        # Output heads (3 classes each: bg, tumor, normal)
        self.psma_out = nn.Conv3d(c, 3, kernel_size=1)
        self.fdg_out = nn.Conv3d(c, 3, kernel_size=1)
        
    def forward(self, psma_input: torch.Tensor, fdg_input: torch.Tensor, 
                cross_attention_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            psma_input: (B, 2, D, H, W) - PSMA PET and CT
            fdg_input: (B, 2, D, H, W) - FDG PET and CT
            cross_attention_weight: Weight for cross-attention (for gradual warmup)
        
        Returns:
            psma_logits: (B, 3, D, H, W) - bg, tumor, normal
            fdg_logits: (B, 3, D, H, W) - bg, tumor, normal
        """
        
        # PSMA Encoding
        p1 = self.psma_enc1(psma_input)
        p2 = self.psma_enc2(p1)
        p3 = self.psma_enc3(p2)
        p4 = self.psma_enc4(p3)
        p_bottleneck = self.psma_bottleneck(p4)
        
        # FDG Encoding
        f1 = self.fdg_enc1(fdg_input)
        f2 = self.fdg_enc2(f1)
        f3 = self.fdg_enc3(f2)
        f4 = self.fdg_enc4(f3)
        f_bottleneck = self.fdg_bottleneck(f4)
        
        # Cross-attention fusion (with gradual warmup via weight)
        if cross_attention_weight > 0:
            # Level 3 cross-attention
            p3_fused = self.cross_attn_3(p3, f3)
            f3_fused = self.cross_attn_3(f3, p3)
            p3 = p3 + cross_attention_weight * (p3_fused - p3)
            f3 = f3 + cross_attention_weight * (f3_fused - f3)
            
            # Level 4 cross-attention
            p4_fused = self.cross_attn_4(p4, f4)
            f4_fused = self.cross_attn_4(f4, p4)
            p4 = p4 + cross_attention_weight * (p4_fused - p4)
            f4 = f4 + cross_attention_weight * (f4_fused - f4)
            
            # Bottleneck cross-attention
            p_bot_fused = self.cross_attn_bottleneck(p_bottleneck, f_bottleneck)
            f_bot_fused = self.cross_attn_bottleneck(f_bottleneck, p_bottleneck)
            p_bottleneck = p_bottleneck + cross_attention_weight * (p_bot_fused - p_bottleneck)
            f_bottleneck = f_bottleneck + cross_attention_weight * (f_bot_fused - f_bottleneck)
        
        # PSMA Decoding
        p_up4 = self.psma_up4(p_bottleneck, p4)
        p_up3 = self.psma_up3(p_up4, p3)
        p_up2 = self.psma_up2(p_up3, p2)
        p_up1 = self.psma_up1(p_up2, p1)
        psma_logits = self.psma_out(p_up1)
        
        # FDG Decoding
        f_up4 = self.fdg_up4(f_bottleneck, f4)
        f_up3 = self.fdg_up3(f_up4, f3)
        f_up2 = self.fdg_up2(f_up3, f2)
        f_up1 = self.fdg_up1(f_up2, f1)
        fdg_logits = self.fdg_out(f_up1)
        
        return psma_logits, fdg_logits


def combined_loss(psma_logits: torch.Tensor, fdg_logits: torch.Tensor,
                  labels: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> dict:
    """
    Compute combined loss for dual-head model
    
    Args:
        psma_logits: (B, 3, D, H, W) - PSMA predictions
        fdg_logits: (B, 3, D, H, W) - FDG predictions  
        labels: (B, D, H, W) - Ground truth with classes 0-4
        class_weights: Optional weights for CE loss
        
    Returns:
        Dictionary with individual and total losses
    """
    
    # Map labels to PSMA classes (0=bg, 1=tumor, 2=normal)
    psma_labels = labels.clone()
    psma_labels[labels == 3] = 0  # FDG tumor -> bg for PSMA
    psma_labels[labels == 4] = 0  # FDG normal -> bg for PSMA
    
    # Map labels to FDG classes (0=bg, 1=tumor, 2=normal)
    fdg_labels = labels.clone()
    fdg_labels[labels == 1] = 0  # PSMA tumor -> bg for FDG
    fdg_labels[labels == 2] = 0  # PSMA normal -> bg for FDG
    fdg_labels[labels == 3] = 1  # FDG tumor
    fdg_labels[labels == 4] = 2  # FDG normal
    
    # Cross-entropy losses
    if class_weights is None:
        class_weights = torch.tensor([0.1, 2.0, 0.5], device=psma_logits.device)
    
    ce_psma = F.cross_entropy(psma_logits, psma_labels, weight=class_weights)
    ce_fdg = F.cross_entropy(fdg_logits, fdg_labels, weight=class_weights)
    
    # Dice losses
    dice_psma = soft_dice_loss(psma_logits, psma_labels)
    dice_fdg = soft_dice_loss(fdg_logits, fdg_labels)
    
    # Combined losses
    psma_loss = 0.5 * ce_psma + 0.5 * dice_psma
    fdg_loss = 0.5 * ce_fdg + 0.5 * dice_fdg
    total_loss = 0.5 * psma_loss + 0.7 * fdg_loss
    
    return {
        'total': total_loss,
        'psma': psma_loss,
        'fdg': fdg_loss,
        'ce_psma': ce_psma,
        'ce_fdg': ce_fdg,
        'dice_psma': dice_psma,
        'dice_fdg': dice_fdg
    }


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for multi-class segmentation
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    # Compute dice per class
    dims = (0, 2, 3, 4)  # Batch and spatial dims
    intersection = torch.sum(probs * targets_one_hot, dim=dims)
    cardinality = torch.sum(probs + targets_one_hot, dim=dims)
    dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)
    
    # Exclude background from loss
    return 1.0 - dice_score[1:].mean()
