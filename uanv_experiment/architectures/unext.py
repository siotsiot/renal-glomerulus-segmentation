"""Architecture-parity copy of the recovered historical UNext implementation.

The implementation intentionally preserves module names, tensor operations,
initialization, additive skips, and optional legacy BRB placement. The BRB is a
residual full-resolution pre-output refinement block, not a deep latent
bottleneck refinement.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, shift_size=5):
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, H, W):
        B, _, C = x.shape
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        _ = num_heads, qkv_bias, qk_scale, attn_drop, sr_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, H, W):
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class BottleneckRefineBlock(nn.Module):
    """Historical 16->8->16 residual full-resolution pre-output refinement."""

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
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[0], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[1], norm_layer=norm_layer, sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        self.refine_bottleneck = (
            BottleneckRefineBlock(in_channels=16, mid_channels=8)
            if use_refine_bottleneck else nn.Identity()
        )
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode="bilinear"))
        out = self.refine_bottleneck(out)
        return self.final(out)
