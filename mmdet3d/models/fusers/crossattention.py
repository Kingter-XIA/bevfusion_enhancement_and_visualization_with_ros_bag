from typing import List, Optional
import torch
from torch import nn
from torch.nn import MultiheadAttention

from mmdet3d.models.builder import FUSERS

__all__ = ["CrossAttentionFuser"]


@FUSERS.register_module()
class CrossAttentionFuser(nn.Module):
    def __init__(
        self,
        in_channels: List[int],  # 输入通道数，例如 [80, 256] 对应 [camera, lidar]
        out_channels: int = 256,  # 输出通道数
        d_model: int = 128,      # 注意力内部维度（小于输入以降低计算量）
        nhead: int = 8,          # 注意力头数
        dropout: float = 0.1,    # Dropout率
        block_size: int = 15,     # 分块大小（平衡计算效率与局部性）
    ) -> None:
        """
        初始化交叉注意力融合模块：
        1. 接收分开的Camera和LiDAR特征，通过交叉注意力融合
        2. 输出形状与ConvFuser一致 (B, out_channels, H, W)
        """
        super().__init__()
        assert len(in_channels) == 2, "输入应为[camera_ch, lidar_ch]"
        self.cam_ch, self.lidar_ch = in_channels
        self.out_channels = out_channels
        self.block_size = block_size

        # 1. 投影层（统一维度到d_model）
        self.cam_proj = nn.Sequential(
            nn.Conv2d(self.cam_ch, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.lidar_proj = nn.Sequential(
            nn.Conv2d(self.lidar_ch, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # 2. 分块交叉注意力
        self.cross_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # 输入输出为 (B, L, C)
        )

        # 3. 输出层（升维到out_channels）
        self.out_conv = nn.Sequential(
            nn.Conv2d(d_model, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播：
        输入: 
            inputs: [camera_feat, lidar_feat], 形状分别为 (B,80,180,180) 和 (B,256,180,180)
        输出: 
            fused_feat: (B,256,180,180)
        """
        camera_feat, lidar_feat = inputs
        B, _, H, W = camera_feat.shape
        
        # 1. 统一维度并分块 ------------------------------------------------
        cam_proj = self.cam_proj(camera_feat)  # (B,d_model,180,180)
        lidar_proj = self.lidar_proj(lidar_feat)

        # 分块 (B,d_model,H,W) -> (B*num_blocks, d_model, block_size, block_size)
        cam_blocks = self._split_into_blocks(cam_proj)  # (B*num_blocks, d_model, block, block)
        lidar_blocks = self._split_into_blocks(lidar_proj)

        # 展平块内空间维度 (B*num_blocks, d_model, block*block)
        cam_blocks = cam_blocks.flatten(2).permute(0, 2, 1)  # (B*num_blocks, block^2, d_model)
        lidar_blocks = lidar_blocks.flatten(2).permute(0, 2, 1)

        # 2. 分块交叉注意力 ------------------------------------------------
        # LiDAR作为Query, Camera作为Key/Value
        attn_out, _ = self.cross_attn(
            query=lidar_blocks,
            key=cam_blocks,
            value=cam_blocks
        )  # (B*num_blocks, block^2, d_model)

        # 3. 合并所有块 ----------------------------------------------------
        attn_out = attn_out.permute(0, 2, 1)  # (B*num_blocks, d_model, block^2)
        attn_out = attn_out.reshape(-1, self.d_model, self.block_size, self.block_size)
        fused = self._merge_blocks(attn_out, B, H, W)  # (B, d_model, 180, 180)

        # 4. 残差连接 + 输出调整 -------------------------------------------
        output = self.out_conv(fused + lidar_proj)  # (B, out_channels, 180, 180)
        return output

    def _split_into_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """将特征图分块 (B,C,H,W) -> (B*num_blocks, C, block, block)"""
        B, C, H, W = x.shape
        x = x.unfold(2, self.block_size, self.block_size)  # (B,C,num_h,block,W)
        x = x.unfold(3, self.block_size, self.block_size)  # (B,C,num_h,num_w,block,block)
        x = x.reshape(B, C, -1, self.block_size, self.block_size)  # (B,C,num_blocks,block,block)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, self.block_size, self.block_size)
        return x

    def _merge_blocks(self, x: torch.Tensor, batch_size: int, H: int, W: int) -> torch.Tensor:
        """合并分块特征 (B*num_blocks,C,block,block) -> (B,C,H,W)"""
        num_blocks = (H // self.block_size) * (W // self.block_size)
        x = x.reshape(batch_size, num_blocks, -1, self.block_size, self.block_size)  # (B,num_blocks,C,block,block)
        x = x.permute(0, 2, 1, 3, 4)  # (B,C,num_blocks,block,block)
        x = x.reshape(batch_size, -1, H // self.block_size, W // self.block_size, self.block_size, self.block_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B,C,num_h,block,num_w,block)
        x = x.reshape(batch_size, -1, H, W)  # (B,C,H,W)
        return x