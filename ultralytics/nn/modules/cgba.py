# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""无人机航拍小目标检测的自定义结构化模块（实验代码）。.

本文件在 YOLO26 上新增两个模块，用于「小目标与中远距目标」的感知增强：

1. ``LBDown`` —— 可学习双边下采样（Learnable Bilateral Downsample）
   替换 backbone 前两级 stride-2 卷积下采样。做法是在 2x2 邻域上取 4 个采样点做
   加权求和实现 1/2 下采样，权重由「空间邻近先验 × 强度相似度」共同决定（双边思想），
   其中：
     - 4 个采样点的亚像素偏移由一个 3x3 卷积预测（可学习，允许偏离固定网格）；
     - 强度响应由 1x1 卷积 + sigmoid 学习，不是固定的灰度差；
     - 相似度带宽 sigma 与 4 个空间先验都是可学习参数。
   动机：常规 stride-2 卷积/平均池化会把仅有几个像素的小目标响应平均掉；双边加权让
   「与中心更相似」的采样点权重更大，从而在降采样的同时保留小目标与边缘的高频响应。

2. ``CGBlockAttn`` —— 粗粒度块注意力（Coarse-Grained Block Attention）
   把特征图按 bs x bs 的不重叠块池化成粗粒度 token（token 数约为 HW/bs^2），只在粗粒度
   token 上做多头自注意力以获得跨区域上下文，再把注意力输出广播回每个像素、经逐像素
   门控后残差相加。动机：以很小的算力代价换取长程上下文，改善小目标在复杂背景下的判别；
   注意力用显式 matmul + softmax 实现（而非 nn.MultiheadAttention），便于 ONNX 导出后
   被 TensorRT 直接解析。

用法（见 ultralytics/cfg/models/26/yolo26-cgba.yaml）：

    - [-1, 1, LBDown, [64]]              # 替换 Conv [64, 3, 2]
    - [-1, 1, CGBlockAttn, [512, 8, 4]]  # 通道数, 块大小, 注意力头数
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .conv import Conv, autopad

__all__ = ("CGBlockAttn", "LBDown")


class LBDown(nn.Module):
    """Learnable Bilateral Downsample：可学习的双边加权 1/2 下采样。.

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k (int): 预测亚像素偏移所用卷积的核大小。
        act (bool | nn.Module): 输出投影后的激活；False 表示恒等。

    Attributes:
        offset (nn.Conv2d): 预测 2x2 窗口内 4 个采样点的亚像素偏移。
        guide (nn.Conv2d): 预测可学习的强度响应（用于双边权重）。
        log_sigma (nn.Parameter): 强度相似度的高斯带宽（对数域）。
        spatial (nn.Parameter): 4 个采样点的空间邻近先验。
        project (Conv): 1x1 输出投影。

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import LBDown
        >>> m = LBDown(3, 16)
        >>> m(torch.randn(1, 3, 64, 64)).shape
        torch.Size([1, 16, 32, 32])
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, act=True):
        """初始化偏移预测、强度响应、双边参数与输出投影。."""
        super().__init__()
        self.offset = nn.Conv2d(c1, 8, k, 1, autopad(k), bias=True)  # 4 点 x 2 方向
        self.offset_scale = 0.5  # 亚像素偏移幅度上限（单位：输入像素）
        self.guide = nn.Conv2d(c1, 1, 1, 1, 0, bias=True)
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.spatial = nn.Parameter(torch.zeros(4))
        self.project = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x):
        """对输入特征图做双边加权的 1/2 下采样。.

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C1, H, W)。

        Returns:
            (torch.Tensor): 输出张量，形状为 (B, C2, ceil(H/2), ceil(W/2))。
        """
        b, _, h, w = x.shape
        ho, wo = (h + 1) // 2, (w + 1) // 2
        # 2x2 窗口的 4 个基准采样点（相对窗口中心，单位：输入像素）
        base = x.new_tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
        # 在窗口中心处取偏移，避免对全分辨率偏移图再采样
        off = torch.tanh(self.offset(x))[:, :, 0::2, 0::2] * self.offset_scale  # (B,8,ho,wo)
        guide = torch.sigmoid(self.guide(x))
        guide_c = guide[:, :, 0::2, 0::2]  # 窗口中心的强度响应
        sigma = F.softplus(self.log_sigma) + 1e-3
        spatial = F.softplus(self.spatial)  # 空间先验保持正值

        yy = torch.arange(ho, device=x.device, dtype=x.dtype) * 2 + 0.5
        xx = torch.arange(wo, device=x.device, dtype=x.dtype) * 2 + 0.5
        cy = yy.view(1, ho, 1).expand(b, ho, wo)
        cx = xx.view(1, 1, wo).expand(b, ho, wo)

        num, den = 0, 0
        for k in range(4):
            py = (cy + base[k, 0] + off[:, 2 * k]) / max(h - 1, 1) * 2 - 1
            px = (cx + base[k, 1] + off[:, 2 * k + 1]) / max(w - 1, 1) * 2 - 1
            grid = torch.stack((px, py), dim=-1)
            # 双边权重 = 可学习空间先验 x 强度相似度（相似则权重大）
            gk = F.grid_sample(guide, grid, mode="bilinear", padding_mode="border", align_corners=True)
            wk = spatial[k] * torch.exp(-((gk - guide_c) ** 2) / (2 * sigma**2))
            sk = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
            num = num + sk * wk
            den = den + wk

        return self.project(num / (den + 1e-6))


class CGBlockAttn(nn.Module):
    """Coarse-Grained Block Attention：块级粗粒度注意力的轻量长程上下文模块。.

    不重叠地按 ``block x block`` 划分特征图，块内平均池化得到粗粒度 token（token 数约为 ``H*W/block^2``），在 token 上做多头自注意力，再广播回像素并用逐像素门控残差相加。
    模块输出通道与输入保持一致，可直接插在 backbone/neck 任意位置。

    Args:
        c1 (int): 输入通道数。
        c2 (int | None): 输出通道数，须与 c1 相同（保证残差）；None 时取 c1。
        block (int): 粗粒度块大小。
        num_heads (int): 注意力头数。
        gate_bias (float): 门控初始偏置，负值使模块初始接近恒等映射。

    Attributes:
        q, k, v (nn.Conv2d): 查询/键/值投影。
        proj (nn.Conv2d): 注意力输出投影。
        gate (nn.Conv2d): 逐像素门控。
        norm (nn.BatchNorm2d): 输出归一化。

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import CGBlockAttn
        >>> m = CGBlockAttn(128, 128, block=8, num_heads=4)
        >>> m(torch.randn(1, 128, 40, 40)).shape
        torch.Size([1, 128, 40, 40])
    """

    def __init__(self, c1, c2=None, block=8, num_heads=4, gate_bias=-2.0):
        """初始化注意力投影、门控与归一化层。."""
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"CGBlockAttn 需要残差结构，要求 c1 == c2，当前 c1={c1}, c2={c2}")
        self.block = int(block)
        self.num_heads = max(int(num_heads), 1)
        hidden = max(c1 // 2, self.num_heads)
        if hidden % self.num_heads:
            hidden = self.num_heads * max(hidden // self.num_heads, 1)
        self.hidden = hidden
        self.q = nn.Conv2d(c1, hidden, 1, 1, 0, bias=True)
        self.k = nn.Conv2d(c1, hidden, 1, 1, 0, bias=True)
        self.v = nn.Conv2d(c1, hidden, 1, 1, 0, bias=True)
        self.proj = nn.Conv2d(hidden, c2, 1, 1, 0, bias=True)
        self.gate = nn.Conv2d(c1, c2, 1, 1, 0, bias=True)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.zeros_(self.proj.bias)

    def _to_tokens(self, x):
        """把 (B,C,H,W) 按不重叠块池化成 (B,C,H/bs*W/bs) 的粗粒度 token。."""
        b, c, h, w = x.shape
        bs = self.block
        pad_h, pad_w = (bs - h % bs) % bs, (bs - w % bs) % bs
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2], x.shape[-1]
        blocks = x.view(b, c, hp // bs, bs, wp // bs, bs).permute(0, 1, 2, 4, 3, 5).reshape(b, c, -1, bs * bs)
        return blocks.mean(dim=-1), (hp, wp)

    def forward(self, x):
        """在粗粒度块 token 上做自注意力，并广播回像素后残差相加。.

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C1, H, W)。

        Returns:
            (torch.Tensor): 输出张量，形状为 (B, C2, H, W)。
        """
        b, _, h, w = x.shape
        bs = self.block
        q, (hp, wp) = self._to_tokens(self.q(x))
        k, _ = self._to_tokens(self.k(x))
        v, _ = self._to_tokens(self.v(x))
        n = q.shape[-1]
        hd = q.shape[1] // self.num_heads
        # (B,heads,N,hd) @ (B,heads,hd,N) -> 粗粒度 token 上的注意力
        q = q.reshape(b, self.num_heads, hd, n).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, hd, n)
        v = v.reshape(b, self.num_heads, hd, n).permute(0, 1, 3, 2)
        attn = torch.softmax(q @ k / math.sqrt(hd), dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(b, self.hidden, n)
        # 广播回像素：同一块内共享注意力结果
        out = out.reshape(b, self.hidden, hp // bs, 1, wp // bs, 1).expand(-1, -1, -1, bs, -1, bs)
        out = out.reshape(b, self.hidden, hp, wp)[:, :, :h, :w]
        y = self.act(self.norm(self.proj(out)))
        return x + y * torch.sigmoid(self.gate(x))
