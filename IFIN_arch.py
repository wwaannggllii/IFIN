import torch
# from torch.functional import Tensor
import torch.nn as nn

from . import SwinT
from torch.nn.modules.activation import Tanhshrink


class SAAB(nn.Module):
    def __init__(self,wn, n_feats):
        super(SAAB, self).__init__()

        self.conv3_1_A = wn(nn.Conv2d(n_feats, n_feats, (3, 1), padding=(1, 0)))
        self.conv3_1_B = wn(nn.Conv2d(n_feats, n_feats, (1, 3), padding=(0, 1)))
        self.adafm = wn(nn.Conv2d(n_feats, n_feats, 3, padding=3 // 2))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.mul_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv3_1_A(x) + self.conv3_1_B(x)
        x1 = self.relu(x1)
        x1 = self.adafm(x1)
        mul = self.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(x1))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(x1)))
        m = x * mul + add
        return m


class ESAB(nn.Module):
    def __init__(self, wn,nf,reduction, use_residual=True, learnable=True):
        super(ESAB, self).__init__()

        self.learnable = learnable
        self.sigmoid = nn.Sigmoid()
        self.norm_layer = wn(nn.Conv2d(nf, nf, 1, 1, 0, bias=True))
        self.conv_1x1 = wn(nn.Conv2d(nf*2, nf, 1, 1, 0, bias=True))

        if self.learnable:
            self.conv_shared = nn.Sequential(wn(nn.Conv2d(nf , nf // reduction, 3, 1, 1, bias=True)),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = wn(nn.Conv2d(nf // reduction, nf, 3, 1, 1, bias=True))
            self.conv_beta = wn(nn.Conv2d(nf // reduction, nf, 3, 1, 1, bias=True))

            self.use_residual = use_residual



            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, loc, glo):
        fusion = self.conv_1x1(torch.cat([loc, glo], dim=1))
        ref_normed_l = self.norm_layer(loc)
        ref_normed_g = self.norm_layer(glo)

        if self.learnable:
            style = self.conv_shared(fusion)
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = fusion.size()
        fusion = fusion.view(b, c, h * w)
        lr_mean = torch.mean(fusion, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(fusion, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean
        out_l = ref_normed_l * self.sigmoid(gamma) + beta

        if self.learnable:
            style_f = self.conv_shared(out_l)
            gamma_f = self.conv_gamma(style_f)
            beta_f = self.conv_beta(style_f)

        b, c, h, w = out_l.size()
        out_l = out_l.view(b, c, h * w)
        lr_mean_f = torch.mean(out_l, dim=-1, keepdim=True).unsqueeze(3)
        lr_std_f = torch.std(out_l, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma_f = gamma_f + lr_std_f
                beta_f = beta_f + lr_mean_f
            else:
                gamma_f = 1 + gamma_f
        else:
            gamma_f = lr_std_f
            beta_f = lr_mean_f

        out = ref_normed_g * self.sigmoid(gamma_f) + beta_f

class IFIN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_recurs, upscale_factor, norm_type=None,
                 act_type='prelu'):
        super(IFIN, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])
        reduction = 5
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

        self.swinT0 = SwinT.SwinT(num_features)
        self.att0 = SAAB(wn, num_features)
        self.rsab0 = ESAB(wn, num_features, reduction)

        self.swinT1 = SwinT.SwinT(num_features)
        self.att1 = SAAB(wn, num_features)
        self.esab1 = ESAB(wn, num_features, reduction)

        self.swinT2 = SwinT.SwinT(num_features)
        self.att2 = SAAB(wn, num_features)
        self.esab2 = ESAB(wn, num_features, reduction)

        self.swinT3 = SwinT.SwinT(num_features)
        self.att3 = SAAB(wn, num_features)
        self.esab3 = ESAB(wn, num_features, reduction)

        self.swinT4 = SwinT.SwinT(num_features)
        self.att4 = SAAB(wn, num_features)
        self.esab4 = ESAB(wn, num_features, reduction)



    def forward(self, x):

        x = (x - self.rgb_mean.cuda() * 255) / 127.5
        s = self.skip(x)
        o0 = self.head(x)

        a0 = self.att0(o0)
        t0 = self.swinT0(o0)
        f0 = self.esab0(a0,t0)

        a1 = self.att1(f0)
        t1 = self.swinT1(f0)
        f1 = self.esab1(a1, t1)

        a2 = self.att2(f1)
        t2 = self.swinT2(f1)
        f2 = self.esab2(a2, t2)

        a3 = self.att3(f2)
        t3 = self.swinT3(f2)
        f3 = self.esab3(a3, t3)

        a4 = self.att4(f3)
        t4 = self.swinT4(f3)
        f4 = self.esab4(a4, t4)

        x = self.tail(f4)

        x += s

        out = x * 127.5 + self.rgb_mean.cuda() * 255
        return out
