# This file is a modified research implementation based in substantial part on:
# UNeXt-pytorch, https://github.com/jeya-maria-jose/UNeXt-pytorch
# Copyright (c) 2022 Jeya Maria Jose — MIT License.
#
# The upstream UNeXt project acknowledges code blocks and helper functions from
# SegFormer, AS-MLP, and pytorch-nested-unet. For SegFormer-derived portions:
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# Those portions are subject to the NVIDIA Source Code License for SegFormer and
# may be used only non-commercially, meaning for research or evaluation purposes.
#
# This file was modified for renal glomerulus binary segmentation, including
# project-specific dataset handling, cross-validation, threshold evaluation,
# bottleneck refinement, and qualitative-analysis support.
#
# Full license texts, affected-component mappings, and additional attributions:
# LICENSE, THIRD_PARTY_NOTICES.md, and LICENSES/.

import argparse
import csv
import glob
import json
import math
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    try:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    except ImportError:
        from torch.nn.init import trunc_normal_

        def to_2tuple(x):
            return (x, x)

        class DropPath(nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    lovasz_hinge = None



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, _, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        _ = num_heads, qkv_bias, qk_scale, attn_drop, sr_ratio


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class BottleneckRefineBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        refined = self.refine(x)
        return self.relu(x + refined)


class UNext(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=3,
        deep_supervision=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=None,
        sr_ratios=None,
        use_refine_bottleneck=False,
        **kwargs,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [128, 160, 256]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depths is None:
            depths = [1, 1, 1]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        _ = deep_supervision, patch_size, in_chans, mlp_ratios, kwargs

        self.encoder1 = nn.Conv2d(input_channels, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])

        ## TODO:학습 돌리기
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.refine_bottleneck = (
            BottleneckRefineBlock(in_channels=16, mid_channels=8)
            if use_refine_bottleneck
            else nn.Identity()
        )
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)


    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out = self.refine_bottleneck(out)

        return self.final(out)


class UNext_S(nn.Module):
    """Conv 3 + MLP 2 + shifted MLP with fewer parameters."""

    def __init__(
        self,
        num_classes,
        input_channels=3,
        deep_supervision=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=None,
        sr_ratios=None,
        use_refine_bottleneck=False,
        **kwargs,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 128, 512]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depths is None:
            depths = [1, 1, 1]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        _ = deep_supervision, patch_size, in_chans, mlp_ratios, kwargs

        self.encoder1 = nn.Conv2d(input_channels, 8, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)

        self.refine_bottleneck = (
            BottleneckRefineBlock(in_channels=8, mid_channels=4)
            if use_refine_bottleneck
            else nn.Identity()
        )
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)


    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)

        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out = self.refine_bottleneck(out)

        return self.final(out)


class VanillaUNet(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=3,
        deep_supervision=False,
        use_refine_bottleneck=False,
        base_channels=64,
        **kwargs,
    ):
        super().__init__()
        _ = deep_supervision, use_refine_bottleneck, kwargs

        self.enc1 = self.conv_block(input_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def match_size(x, skip):
        if x.shape[-2:] == skip.shape[-2:]:
            return x
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.match_size(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.match_size(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)



class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + self.smooth) / (input.sum(1) + target.sum(1) + self.smooth)
        dice = 1 - dice.sum() / num
        return self.bce_weight * bce + self.dice_weight * dice


class BCEDiceLoss(WeightedBCEDiceLoss):
    def __init__(self, bce_weight=0.5, dice_weight=1.0):
        super().__init__(bce_weight=bce_weight, dice_weight=dice_weight)


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if lovasz_hinge is None:
            raise ImportError("LovaszSoftmax is not installed. Install it to use LovaszHingeLoss.")
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

ARCHS = {
    "UNext": UNext,
    "UNext_S": UNext_S,
    "UNet": VanillaUNet,
}

LOSSES = {
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "BCEDiceLoss": BCEDiceLoss,
    "WeightedBCEDiceLoss": WeightedBCEDiceLoss,
    "LovaszHingeLoss": LovaszHingeLoss,
}
LOSS_NAMES = tuple(LOSSES.keys())


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f"}:
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def count_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def iou_score(output, target, smooth=1e-5):
    if output.shape[1] > 1:
        output = torch.argmax(output, dim=1)
        if target.ndim == 4 and target.shape[1] > 1:
            target = torch.argmax(target, dim=1)
        elif target.ndim == 4:
            target = target.squeeze(1)
        output = output.float()
        target = target.float()
    else:
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = (target > 0.5).float()

    intersection = (output * target).sum()
    union = output.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return iou.item(), dice.item()


class SegmentationDataset(Dataset):
    def __init__(
        self,
        img_ids,
        img_dir,
        mask_dir,
        img_ext=".png",
        mask_ext=".png",
        num_classes=1,
        input_h=800,
        input_w=800,
        augment=False,
    ):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.input_h = input_h
        self.input_w = input_w
        self.augment = augment

    def __len__(self):
        return len(self.img_ids)

    def _read_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        original_shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA
        if image.shape[0] < self.input_h or image.shape[1] < self.input_w:
            interpolation = cv2.INTER_LINEAR
        image = cv2.resize(image, (self.input_w, self.input_h), interpolation=interpolation)
        image = image.astype(np.float32) / 255.0
        return np.transpose(image, (2, 0, 1)), original_shape

    def _read_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        original_shape = mask.shape
        mask = cv2.resize(mask, (self.input_w, self.input_h), interpolation=cv2.INTER_NEAREST)

        if self.num_classes <= 1:
            mask = (mask > 0).astype(np.float32)[None, ...]
            return mask, original_shape

        one_hot_mask = np.zeros((self.num_classes, self.input_h, self.input_w), dtype=np.float32)
        for class_index in range(self.num_classes):
            one_hot_mask[class_index] = (mask == class_index).astype(np.float32)
        return one_hot_mask, original_shape

    def _augment_pair(self, image, mask):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=2)

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=1)

        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(1, 2))

        if np.random.rand() < 0.8:
            contrast = np.random.uniform(0.9, 1.1)
            brightness = np.random.uniform(-0.05, 0.05)
            image = np.clip(image * contrast + brightness, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}{self.img_ext}")
        mask_path = os.path.join(self.mask_dir, f"{img_id}{self.mask_ext}")

        image, original_image_shape = self._read_image(img_path)
        mask, original_mask_shape = self._read_mask(mask_path)
        if self.augment:
            image = np.transpose(image, (1, 2, 0))
            image, mask = self._augment_pair(image, mask)
            image = np.transpose(image, (2, 0, 1))

        foreground_ratio = float((mask > 0).mean())

        meta = {
            "id": img_id,
            "image_path": img_path,
            "mask_path": mask_path,
            "original_image_shape": str(tuple(original_image_shape)),
            "original_mask_shape": str(tuple(original_mask_shape)),
            "foreground_ratio": foreground_ratio,
        }

        return torch.from_numpy(image), torch.from_numpy(mask), meta


def normalize_ext(ext):
    ext = str(ext).strip()
    if not ext:
        raise ValueError("file extension must not be empty")
    return ext if ext.startswith(".") else f".{ext}"


def collect_stems(data_dir, ext):
    pattern = os.path.join(data_dir, f"*{ext}")
    return {os.path.splitext(os.path.basename(path))[0] for path in glob.glob(pattern)}


def mask_has_foreground(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return bool(np.any(mask > 0))


def collect_paired_ids(img_dir, mask_dir, img_ext, mask_ext, exclude_empty_masks=False):
    img_ids = collect_stems(img_dir, img_ext)
    mask_ids = collect_stems(mask_dir, mask_ext)
    paired_ids = sorted(img_ids & mask_ids)
    missing_mask_ids = sorted(img_ids - mask_ids)
    extra_mask_ids = sorted(mask_ids - img_ids)
    excluded_empty_ids = []

    if exclude_empty_masks:
        kept_ids = []
        for img_id in paired_ids:
            mask_path = os.path.join(mask_dir, f"{img_id}{mask_ext}")
            if mask_has_foreground(mask_path):
                kept_ids.append(img_id)
            else:
                excluded_empty_ids.append(img_id)
        paired_ids = kept_ids

    return {
        "image_count": len(img_ids),
        "mask_count": len(mask_ids),
        "paired_ids": paired_ids,
        "missing_mask_ids": missing_mask_ids,
        "extra_mask_ids": extra_mask_ids,
        "excluded_empty_ids": excluded_empty_ids,
    }


def split_paired_ids(paired_ids, val_ratio, seed):
    if not paired_ids:
        raise RuntimeError("No paired image/mask ids found.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    shuffled = list(paired_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])

    if not train_ids:
        raise RuntimeError("Train split is empty. Use a smaller --val_ratio.")
    return train_ids, val_ids


def write_id_list(csv_path, ids):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id"])
        for img_id in ids:
            writer.writerow([img_id])


def first_meta_value(meta, key):
    value = meta[key]
    if isinstance(value, (list, tuple)):
        return value[0]
    if torch.is_tensor(value):
        return value[0].item()
    return value


def print_first_batch_debug(images, targets, meta, fold_label=None):
    first_mask = targets[0].detach().cpu()
    mask_unique = sorted(float(value) for value in torch.unique(targets.detach().cpu()))
    foreground_ratio = float((first_mask > 0).float().mean().item())

    prefix = f"[{fold_label}] " if fold_label else ""
    print(f"{prefix}Debug first batch sample")
    print(f"  image_path: {first_meta_value(meta, 'image_path')}")
    print(f"  mask_path: {first_meta_value(meta, 'mask_path')}")
    print(f"  original_image_shape: {first_meta_value(meta, 'original_image_shape')}")
    print(f"  original_mask_shape: {first_meta_value(meta, 'original_mask_shape')}")
    print(f"  resized_image_tensor_shape: {tuple(images[0].shape)}")
    print(f"  resized_mask_tensor_shape: {tuple(targets[0].shape)}")
    print(f"  processed_mask_unique_values: {mask_unique}")
    print(f"  foreground_ratio: {foreground_ratio:.10f}")
    if not set(mask_unique).issubset({0.0, 1.0}):
        print("  warning: processed mask has values outside {0, 1}")


def tensor_image_to_bgr(image_tensor):
    image = image_tensor.detach().cpu().numpy()
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    image = np.transpose(image[:3], (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def output_to_binary_mask(output_tensor, threshold=0.5):
    output = output_tensor.detach().cpu()
    if output.ndim == 3 and output.shape[0] > 1:
        return (torch.argmax(output, dim=0).numpy() > 0).astype(np.uint8)
    if output.ndim == 3:
        output = output[0]
    probs = torch.sigmoid(output)
    return (probs.numpy() > threshold).astype(np.uint8)


def target_to_binary_mask(target_tensor):
    target = target_tensor.detach().cpu()
    if target.ndim == 3 and target.shape[0] > 1:
        return (torch.argmax(target, dim=0).numpy() > 0).astype(np.uint8)
    if target.ndim == 3:
        target = target[0]
    return (target.numpy() > 0.5).astype(np.uint8)


def save_prediction_overlay(image_tensor, target_tensor, output_tensor, output_path, threshold=0.5):
    image_bgr = tensor_image_to_bgr(image_tensor)
    gt = target_to_binary_mask(target_tensor)
    pred = output_to_binary_mask(output_tensor, threshold=threshold)

    gt_panel = np.zeros_like(image_bgr)
    gt_panel[:, :, 1] = gt * 255

    pred_panel = np.zeros_like(image_bgr)
    pred_panel[:, :, 2] = pred * 255

    overlay = image_bgr.copy()
    gt_pixels = gt.astype(bool)
    pred_pixels = pred.astype(bool)
    overlay[gt_pixels] = cv2.addWeighted(overlay, 0.55, gt_panel, 0.45, 0)[gt_pixels]
    overlay[pred_pixels] = cv2.addWeighted(overlay, 0.55, pred_panel, 0.45, 0)[pred_pixels]

    panel = np.concatenate([image_bgr, gt_panel, pred_panel, overlay], axis=1)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, panel)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="model name")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=16, type=int, metavar="N", help="mini-batch size")

    parser.add_argument("--arch", "-a", metavar="ARCH", default="UNext", choices=tuple(ARCHS.keys()))
    parser.add_argument("--deep_supervision", default=False, type=str2bool)
    parser.add_argument("--input_channels", default=3, type=int, help="input channels")
    parser.add_argument("--num_classes", default=1, type=int, help="number of classes")
    parser.add_argument("--input_w", default=800, type=int, help="resized image width")
    parser.add_argument("--input_h", default=800, type=int, help="resized image height")
    parser.add_argument("--img_size", default=None, type=int, help="optional square resize size; overrides input_w/input_h")
    parser.add_argument(
        "--use_refine_bottleneck",
        action="store_true",
        help="enable optional bottleneck refinement block before final output conv",
    )

    parser.add_argument("--loss", default="BCEDiceLoss", choices=LOSS_NAMES)
    parser.add_argument("--bce_weight", default=0.5, type=float, help="BCE weight for WeightedBCEDiceLoss")
    parser.add_argument("--dice_weight", default=1.0, type=float, help="Dice loss weight for WeightedBCEDiceLoss")

    parser.add_argument("--dataset", default="glomerulus", help="dataset name")
    parser.add_argument("--img_ext", default=".png", help="image file extension")
    parser.add_argument("--mask_ext", default=".png", help="mask file extension")
    parser.add_argument("--img_dir", default="data/images", type=str, help="directory containing all images")
    parser.add_argument("--mask_dir", default="data/masks", type=str, help="directory containing provided masks")
    parser.add_argument("--val_ratio", default=0.2, type=float, help="validation split ratio from paired ids")
    parser.add_argument("--seed", default=42, type=int, help="random seed for paired train/val split")
    parser.add_argument("--kfolds", default=0, type=int, help="number of KFold splits; values <2 use val_ratio split")
    parser.add_argument("--fold", default=0, type=int, help="1-based fold number to run when kfolds >= 2; 0 means fold 1")
    parser.add_argument("--run_all_folds", action="store_true", help="run all folds sequentially when kfolds >= 2")
    parser.add_argument("--exclude_empty_masks", action="store_true", help="exclude paired masks with no foreground")
    parser.add_argument("--aug", action="store_true", help="enable safe train-only augmentation")
    parser.add_argument("--overlay_count", default=20, type=int, help="number of validation overlays to save per epoch")
    parser.add_argument("--overlay_threshold", default=0.5, type=float, help="threshold for binary prediction overlays")
    parser.add_argument(
        "--checkpoint_metric",
        default="val_dice",
        choices=["val_dice", "val_iou"],
        help="validation metric used to save the best checkpoint",
    )

    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", "--learning_rate", default=1e-3, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--nesterov", default=False, type=str2bool, help="nesterov")

    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=["CosineAnnealingLR", "ReduceLROnPlateau", "MultiStepLR", "ConstantLR"],
    )
    parser.add_argument("--min_lr", default=1e-5, type=float, help="minimum learning rate")
    parser.add_argument("--factor", default=0.1, type=float)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--milestones", default="1,2", type=str)
    parser.add_argument("--gamma", default=2 / 3, type=float)
    parser.add_argument("--early_stopping", default=-1, type=int, metavar="N", help="early stopping")

    parser.add_argument("--num_workers", default=4, type=int)
    return parser.parse_args()


def train(config, train_loader, model, criterion, optimizer, device, fold_label=None):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

    model.train()
    first_batch_debug_printed = False
    desc = f"{fold_label} train" if fold_label else "train"
    pbar = tqdm(total=len(train_loader), desc=desc, leave=False)
    for images, targets, meta in train_loader:
        if not first_batch_debug_printed:
            print_first_batch_debug(images, targets, meta, fold_label=fold_label)
            first_batch_debug_printed = True

        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")

        if config["deep_supervision"]:
            outputs = model(images)
            loss = 0.0
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)
            iou, _ = iou_score(outputs[-1], targets)
        else:
            output = model(images)
            loss = criterion(output, targets)
            iou, _ = iou_score(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters["loss"].update(loss.item(), images.size(0))
        avg_meters["iou"].update(iou, images.size(0))

        pbar.set_postfix(OrderedDict([("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)])


def validate(config, val_loader, model, criterion, device, epoch, overlay_dir, fold_label=None):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter(), "dice": AverageMeter()}

    model.eval()
    saved_overlays = 0
    with torch.no_grad():
        desc = f"{fold_label} val" if fold_label else "val"
        pbar = tqdm(total=len(val_loader), desc=desc, leave=False)
        for images, targets, meta in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")

            if config["deep_supervision"]:
                outputs = model(images)
                loss = 0.0
                for output in outputs:
                    loss += criterion(output, targets)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], targets)
            else:
                output = model(images)
                loss = criterion(output, targets)
                iou, dice = iou_score(output, targets)

            avg_meters["loss"].update(loss.item(), images.size(0))
            avg_meters["iou"].update(iou, images.size(0))
            avg_meters["dice"].update(dice, images.size(0))

            overlay_output = outputs[-1] if config["deep_supervision"] else output
            if config["overlay_count"] > 0 and saved_overlays < config["overlay_count"]:
                batch_ids = meta["id"]
                for sample_index in range(images.size(0)):
                    if saved_overlays >= config["overlay_count"]:
                        break
                    sample_id = batch_ids[sample_index] if isinstance(batch_ids, (list, tuple)) else str(saved_overlays)
                    overlay_path = os.path.join(
                        overlay_dir,
                        f"epoch_{epoch:03d}",
                        f"{saved_overlays + 1:02d}_{sample_id}.png",
                    )
                    save_prediction_overlay(
                        images[sample_index],
                        targets[sample_index],
                        overlay_output[sample_index],
                        overlay_path,
                        threshold=config["overlay_threshold"],
                    )
                    saved_overlays += 1

            pbar.set_postfix(
                OrderedDict(
                    [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg), ("dice", avg_meters["dice"].avg)]
                )
            )
            pbar.update(1)
        pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg), ("dice", avg_meters["dice"].avg)]
    )


def save_log_to_csv(log, csv_path):
    fieldnames = list(log.keys())
    row_count = len(log[fieldnames[0]])

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(row_count):
            writer.writerow({field: log[field][row_index] for field in fieldnames})


def build_kfold_splits(paired_ids, kfolds, seed):
    if kfolds < 2:
        raise ValueError("--kfolds must be at least 2 to use KFold.")
    if len(paired_ids) < kfolds:
        raise ValueError(f"kfolds={kfolds} is larger than paired_ids_count={len(paired_ids)}.")

    try:
        from sklearn.model_selection import KFold
    except ImportError as exc:
        raise ImportError("scikit-learn is required for --kfolds >= 2. Install sklearn and retry.") from exc

    ids = np.array(sorted(paired_ids))
    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    splits = []
    for fold_index, (train_indices, val_indices) in enumerate(kfold.split(ids), start=1):
        train_ids = sorted(ids[train_indices].tolist())
        val_ids = sorted(ids[val_indices].tolist())
        splits.append((fold_index, train_ids, val_ids))
    return splits


def select_kfold_splits(splits, fold, run_all_folds):
    if run_all_folds:
        return splits

    selected_fold = 1 if fold == 0 else fold
    if selected_fold < 1 or selected_fold > len(splits):
        raise ValueError(f"--fold must be between 1 and {len(splits)} for this run, or 0 for fold 1.")

    return [splits[selected_fold - 1]]


def make_model(config, device):
    model_kwargs = {}
    if config["input_h"] == config["input_w"]:
        model_kwargs["img_size"] = config["input_h"]

    model = ARCHS[config["arch"]](
        config["num_classes"],
        config["input_channels"],
        config["deep_supervision"],
        use_refine_bottleneck=config.get("use_refine_bottleneck", False),
        **model_kwargs,
    ).to(device)
    return model


def make_optimizer(config, model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"] == "Adam":
        return optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "SGD":
        return optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
            weight_decay=config["weight_decay"],
        )
    raise NotImplementedError


def make_scheduler(config, optimizer):
    if config["scheduler"] == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["min_lr"])
    if config["scheduler"] == "ReduceLROnPlateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            verbose=True,
            min_lr=config["min_lr"],
        )
    if config["scheduler"] == "MultiStepLR":
        milestones = [int(epoch) for epoch in config["milestones"].split(",") if epoch]
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=config["gamma"])
    if config["scheduler"] == "ConstantLR":
        return None
    raise NotImplementedError


def make_criterion(config, device):
    if config["loss"] in {"BCEDiceLoss", "WeightedBCEDiceLoss"}:
        return LOSSES[config["loss"]](
            bce_weight=config["bce_weight"],
            dice_weight=config["dice_weight"],
        ).to(device)
    return LOSSES[config["loss"]]().to(device)


def write_kfold_summary(summary_rows, csv_path):
    fieldnames = [
        "fold",
        "train_count",
        "val_count",
        "best_epoch",
        "best_val_loss",
        "best_val_iou",
        "best_val_dice",
        "best_checkpoint_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def write_kfold_mean_std(summary_rows, csv_path):
    def metric_values(name):
        return np.array([float(row[name]) for row in summary_rows], dtype=np.float64)

    dice = metric_values("best_val_dice")
    iou = metric_values("best_val_iou")
    loss = metric_values("best_val_loss")
    row = {
        "val_dice_mean": float(np.mean(dice)),
        "val_dice_std": float(np.std(dice, ddof=1)) if len(dice) > 1 else 0.0,
        "val_iou_mean": float(np.mean(iou)),
        "val_iou_std": float(np.std(iou, ddof=1)) if len(iou) > 1 else 0.0,
        "val_loss_mean": float(np.mean(loss)),
        "val_loss_std": float(np.std(loss, ddof=1)) if len(loss) > 1 else 0.0,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def train_one_fold(config, model_dir, paired_info, train_img_ids, val_img_ids, fold_index=None, fold_count=None):
    fold_label = f"Fold {fold_index}/{fold_count}" if fold_index is not None and fold_count is not None else None
    overlay_dir = os.path.join(model_dir, "val_overlays")
    os.makedirs(model_dir, exist_ok=True)

    fold_config = dict(config)
    fold_config["active_fold"] = fold_index
    fold_config["fold_count"] = fold_count
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as config_file:
        json.dump(fold_config, config_file, indent=2, ensure_ascii=False)

    write_id_list(os.path.join(model_dir, "paired_ids.csv"), paired_info["paired_ids"])
    write_id_list(os.path.join(model_dir, "missing_masks.csv"), paired_info["missing_mask_ids"])
    write_id_list(os.path.join(model_dir, "extra_masks.csv"), paired_info["extra_mask_ids"])
    write_id_list(os.path.join(model_dir, "excluded_empty_masks.csv"), paired_info["excluded_empty_ids"])
    write_id_list(os.path.join(model_dir, "train_ids.csv"), train_img_ids)
    write_id_list(os.path.join(model_dir, "val_ids.csv"), val_img_ids)

    if fold_label:
        print(f"[{fold_label}] model_dir: {model_dir}")
        print(f"[{fold_label}] train_count: {len(train_img_ids)}, val_count: {len(val_img_ids)}")
    else:
        print("[Data] train_count:", len(train_img_ids))
        print("[Data] val_count:", len(val_img_ids))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = device.type == "cuda"

    criterion = make_criterion(config, device)
    model = make_model(config, device)
    print(f"Model: {config['arch']} | Params: {count_params(model):,}")
    optimizer = make_optimizer(config, model)
    scheduler = make_scheduler(config, optimizer)

    train_dataset = SegmentationDataset(
        img_ids=train_img_ids,
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
        augment=config["aug"],
    )
    val_dataset = SegmentationDataset(
        img_ids=val_img_ids,
        img_dir=config["img_dir"],
        mask_dir=config["mask_dir"],
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
            ("val_dice", []),
        ]
    )

    best_score = -float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_iou = 0.0
    best_val_dice = 0.0
    best_checkpoint_path = os.path.join(model_dir, "best_checkpoint.pth")
    trigger = 0

    for epoch in range(1, config["epochs"] + 1):
        epoch_prefix = f"[{fold_label}] " if fold_label else ""
        print(f"{epoch_prefix}Epoch [{epoch}/{config['epochs']}]")

        train_log = train(config, train_loader, model, criterion, optimizer, device, fold_label=fold_label)
        val_log = validate(config, val_loader, model, criterion, device, epoch, overlay_dir, fold_label=fold_label)

        if scheduler is not None:
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_log["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch_prefix}loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f"
            % (train_log["loss"], train_log["iou"], val_log["loss"], val_log["iou"], val_log["dice"])
        )

        log["epoch"].append(epoch)
        log["lr"].append(current_lr)
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(val_log["loss"])
        log["val_iou"].append(val_log["iou"])
        log["val_dice"].append(val_log["dice"])
        save_log_to_csv(log, os.path.join(model_dir, "log.csv"))

        trigger += 1
        current_score = val_log["dice"] if config["checkpoint_metric"] == "val_dice" else val_log["iou"]
        if current_score > best_score:
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            torch.save(
                {
                    "epoch": epoch,
                    "config": fold_config,
                    "checkpoint_metric": config["checkpoint_metric"],
                    "best_score": current_score,
                    "best_val_loss": val_log["loss"],
                    "best_val_iou": val_log["iou"],
                    "best_val_dice": val_log["dice"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_checkpoint_path,
            )
            best_score = current_score
            best_epoch = epoch
            best_val_loss = val_log["loss"]
            best_val_iou = val_log["iou"]
            best_val_dice = val_log["dice"]
            print(f"{epoch_prefix}=> saved best model ({config['checkpoint_metric']}={best_score:.6f})")
            trigger = 0

        if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
            print(f"{epoch_prefix}=> early stopping")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "fold": fold_index if fold_index is not None else "single",
        "train_count": len(train_img_ids),
        "val_count": len(val_img_ids),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "best_val_dice": best_val_dice,
        "best_checkpoint_path": best_checkpoint_path,
    }


def main():
    config = vars(parse_args())

    if config["img_size"] is not None:
        config["input_h"] = config["img_size"]
        config["input_w"] = config["img_size"]
    config["img_ext"] = normalize_ext(config["img_ext"])
    config["mask_ext"] = normalize_ext(config["mask_ext"])

    if not config["name"]:
        suffix = "wDS" if config["deep_supervision"] else "woDS"
        config["name"] = f"{config['dataset']}_{config['arch']}_{suffix}"

    if config["input_h"] % 32 != 0 or config["input_w"] % 32 != 0:
        print(
            "[Warning] UNext down/up-sampling is safest when input_h and input_w are divisible by 32. "
            f"Got input_h={config['input_h']}, input_w={config['input_w']}."
        )

    print("-" * 20)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 20)

    base_model_dir = os.path.join("models", config["name"])
    os.makedirs(base_model_dir, exist_ok=True)

    paired_info = collect_paired_ids(
        config["img_dir"],
        config["mask_dir"],
        config["img_ext"],
        config["mask_ext"],
        exclude_empty_masks=config["exclude_empty_masks"],
    )
    paired_ids = paired_info["paired_ids"]

    print("[Data] image_count:", paired_info["image_count"])
    print("[Data] mask_count:", paired_info["mask_count"])
    print("[Data] paired_ids_count:", len(paired_ids))
    print("[Data] missing_masks_count:", len(paired_info["missing_mask_ids"]))
    print("[Data] extra_masks_count:", len(paired_info["extra_mask_ids"]))
    print("[Data] excluded_empty_masks_count:", len(paired_info["excluded_empty_ids"]))

    if config["kfolds"] >= 2:
        splits = build_kfold_splits(paired_ids, config["kfolds"], config["seed"])
        selected_splits = select_kfold_splits(splits, config["fold"], config["run_all_folds"])
        summary_rows = []

        for fold_index, train_img_ids, val_img_ids in selected_splits:
            fold_model_dir = os.path.join(base_model_dir, f"fold_{fold_index}")
            summary_row = train_one_fold(
                config,
                fold_model_dir,
                paired_info,
                train_img_ids,
                val_img_ids,
                fold_index=fold_index,
                fold_count=config["kfolds"],
            )
            summary_rows.append(summary_row)

        write_kfold_summary(summary_rows, os.path.join(base_model_dir, "kfold_summary.csv"))
        write_kfold_mean_std(summary_rows, os.path.join(base_model_dir, "kfold_mean_std.csv"))
        print(f"[KFold] summary saved to {os.path.join(base_model_dir, 'kfold_summary.csv')}")
        print(f"[KFold] mean/std saved to {os.path.join(base_model_dir, 'kfold_mean_std.csv')}")
        return

    if config["run_all_folds"]:
        raise ValueError("--run_all_folds requires --kfolds >= 2.")

    train_img_ids, val_img_ids = split_paired_ids(paired_ids, config["val_ratio"], config["seed"])
    train_one_fold(config, base_model_dir, paired_info, train_img_ids, val_img_ids)


if __name__ == "__main__":
    main()
