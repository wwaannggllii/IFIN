import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F
from einops import rearrange
class SAAB(nn.Module):
    def __init__(self, num_features, use_spatial=True, use_channel=True, \
                 cha_ratio=1, spa_ratio=1, down_ratio=1):
        super(SAAB, self).__init__()

        self.in_channel = num_features
        self.in_spatial = num_features

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = num_features // cha_ratio
        self.inter_spatial = num_features // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            # num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2+1, out_channels=self.inter_channel // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(self.inter_channel // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.inter_channel // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )


    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1) #
            phi_xs = phi_xs.view(b, self.inter_channel, -1) #
            Gs = torch.matmul(theta_xs, phi_xs) # h * w  h * w
            Gs_in = Gs.permute(0, 2, 1)
            Gs_out = Gs
            g_xs = self.gx_spatial(x)
            g_xs1 =g_xs.view(b, -1, h * w)
            g_xs2 = torch.mean(g_xs, dim=1, keepdim=True)

            Gs_in1 = torch.matmul(g_xs1, Gs_in).view(b, c, h, w)
            Gs_out1 = torch.matmul(g_xs1, Gs_out).view(b, c, h, w)
            Gs_joint = torch.cat((Gs_in1, Gs_out1,g_xs2), 1)
            W_ys = self.W_spatial(Gs_joint)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                out = F.sigmoid(W_ys.expand_as(x)) * x
            return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos



class Attention(nn.Module):
    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.mlp = Mlp(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.FeaExtract = FeaExtract(dim)

    def forward(self, x, mask=None):
        o0 = self.FeaExtract(x)
        UP_LRHSI = o0.clamp_(0, 1)
        sz = UP_LRHSI.size(2)
        E = rearrange(o0, 'B c H W -> B (H W) c', H=sz)
        B_, N, C = E.shape

        qkv = self.qk(E).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(E).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases)  # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) + E
        x_proj = self.proj(x)
        x = self.mlp(x_proj) + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3,
                                              dilation=3, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=5,
                                              dilation=5, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel * 3, channel, kernel_size=1, stride=1, padding=0, groups=channel,bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return buffer_1 + x

class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv3_1_A = nn.Conv2d(channel, channel, (3, 1), padding=(1, 0))
        self.conv3_1_B = nn.Conv2d(channel, channel, (1, 3), padding=(0, 1))
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        buffer = self.conv3_1_A(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv3_1_B(buffer)
        return buffer + x

class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = RB(channel)

    def forward(self, x):
        buffer_x = self.FERB_1(x)
        buffer_x = self.FERB_2(buffer_x)
        return buffer_x

class MSTB(nn.Module):

    def __init__(self,dim, num_heads=4, group_size=1, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.group_size = group_size
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        # if min(self.input_resolution) <= self.group_size:
        #     # if group size is larger than input resolution, we don't partition groups
        #     self.lsda_flag = 0
        #     self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        self.patch_embed = PatchEmbed(
            embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x):
        _, _, H, W = x.size()
        x_size = (H, W)
        x0 = self.patch_embed(x)  # [B, C, H, W]-> [B, HW, C]
        shortcut = x0

        B, L, C = x0.shape
        # multi-head self-attention
        x = self.attn(x, mask=self.attn_mask)  # nW*B, G*G, C
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.patch_unembed(x, x_size)
        return x


class IFIN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu', reduction=1):
        super(IFIN, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])


        head = []
        head.append(
            wn(nn.Conv2d(in_channels, num_features, 3, padding=3 // 2)))

        out_feats = upscale_factor * upscale_factor * out_channels
        tail = []
        tail.append(wn(nn.Conv2d(num_features, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(upscale_factor))

        skip = []
        skip.append(wn(nn.Conv2d(in_channels, out_feats, 5, padding=5 // 2)))
        skip.append(nn.PixelShuffle(upscale_factor))

        self.head = nn.Sequential(*head)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        self.mstb0 = MSTB(num_features)
        self.mstb1 = MSTB(num_features)
        self.mstb2 = MSTB(num_features)
        self.mstb3 = MSTB(num_features)
        self.saab0 = SAAB(num_features)
        self.saab1 = SAAB(num_features)
        self.saab2 = SAAB(num_features)
        self.saab3 = SAAB(num_features)


    def forward(self, x):

        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        s = self.skip(x)
        o0 = self.head(x)

        a0 = self.saab0(o0)
        b0 = self.mstb0(a0)

        a1 = self.saab1(b0)
        b1 = self.mstb1(a1)

        a2 = self.saab2(b1)
        b2 = self.mstb2(a2)

        a3 = self.saab3(b2)
        b3 = self.mstb3(a3)

        x = self.tail(b3)

        x += s

        out = x * 127.5 + self.rgb_mean.cuda() * 255
        return out
