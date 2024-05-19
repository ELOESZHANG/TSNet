import torch
import torch.nn as nn
from basicseg.utils.registry import NET_REGISTRY

##########################################################################
# same conv（输入输出的size不变，channel由in_channels/out_channels决定）
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class PMSFEM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention =Attention(dim)
        self.dconv1 = nn.Conv2d(dim, dim, 3, stride=1, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(dim, dim, 3, stride=1, padding=3, dilation=3)
        self.conv1 = conv(3*dim, dim, 1)
    def forward(self, x):
        x1 = self.self_attention(x)
        x2 = self.dconv1(x)
        x3 = self.dconv2(x)
        concatenated_tensor = torch.cat((x1,x2,x3), dim=1)
        res = self.conv1(concatenated_tensor)+x
        return res
##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, 1)
        self.conv2 = conv(n_feat, n_feat, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.sigmoid(self.conv2(x1))
        return x*x2

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]
        self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in
                               range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels// 2, out_channels=out_channels// 2, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels// 2),
            torch.nn.ReLU(inplace=True)
        )
        self.conv3 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        output = self.conv2(x)
        output = self.conv3(x + output)
        return output


class ConvResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.75, ratio_gout=0.75, padding=0, bias=False,
                 padding_type='reflect'):
        super(ConvResBlock, self).__init__()
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        in_cg = int(in_channels * self.ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * self.ratio_gout)
        out_cl = out_channels - out_cg
        self.global_in_num = in_cg
        self.convl2l = nn.Conv2d(in_channels=in_cl, out_channels=out_cl ,kernel_size=kernel_size,
                              padding=padding, bias=bias, padding_mode=padding_type)
        self.convl2g = nn.Conv2d(in_channels=in_cl, out_channels=out_cg, kernel_size=kernel_size,
                              padding=padding, bias=bias, padding_mode=padding_type)
        self.convg2l = nn.Conv2d(in_channels=in_cg, out_channels=out_cl, kernel_size=kernel_size,
                              padding=padding, bias=bias, padding_mode=padding_type)
        self.convg2g = SpectralTransform(in_cg, out_cg)
    def forward(self, x):
        x_l, x_g = x # x is tuple
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class CFM(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0.75, ratio_gout=0.75,padding=0, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect'):
        super(CFM, self).__init__()

        self.crb = ConvResBlock(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, padding,bias,padding_type=padding_type,)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
    def forward(self, x):
        x_l, x_g = self.crb(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


#   Fast Fourier Conv Residual Block
class CFMBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, outline=False):
        super().__init__()
        self.conv1 = CFM(dim, dim, kernel_size=3, padding=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type)
        self.conv2 = CFM(dim, dim, kernel_size=3, padding=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type)
        self.inline = inline
        self.outline = outline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.crb.global_in_num], x[:, -self.conv1.crb.global_in_num:]
        else:
            x_l, x_g = x
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.outline:
            # 在给定维度上对输入的张量序列seq进行连接操作
            out = torch.cat(out, dim=1)
        return out


##########################################################################
class SNet(nn.Module):
    def __init__(self, n_feat):
        super(SNet, self).__init__()
        net = nn.ModuleList()
        cur_resblock1 = CFMBlock(n_feat, padding_type='reflect', activation_layer=nn.ReLU,
                                      norm_layer=nn.BatchNorm2d, inline=True, outline=False)
        net.append(cur_resblock1)
        cur_resblock2 = CFMBlock(n_feat, padding_type='reflect', activation_layer=nn.ReLU,
                                              norm_layer=nn.BatchNorm2d, inline=False, outline=True)
        net.append(cur_resblock2)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


@NET_REGISTRY.register()
class TSISSFNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, n_feat=10, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(TSISSFNet, self).__init__()
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = SNet(n_feat + scale_orsnetfeats)

        self.sam12 = SAM(n_feat)
        self.sam23 = SAM(n_feat)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

        self.last_conv = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1)

        self.conv1 = nn.Conv2d(n_feat, out_c, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(n_feat, out_c, kernel_size=1, stride=1)

        self.att1 = PMSFEM(n_feat + scale_unetfeats * 2)
        self.att2 = PMSFEM(n_feat + scale_unetfeats * 2)
    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img = x3_img[:, :, 0:int(H / 2), :]
        x2bot_img = x3_img[:, :, int(H / 2):H, :]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:, :, :, 0:int(W / 2)]
        x1rtop_img = x2top_img[:, :, :, int(W / 2):W]
        x1lbot_img = x2bot_img[:, :, :, 0:int(W / 2)]
        x1rbot_img = x2bot_img[:, :, :, int(W / 2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        for i in range(len(feat1_ltop)):
            if i == 2:
                feat1_ltop[i] = self.att1(feat1_ltop[i])
                feat1_rtop[i] = self.att1(feat1_rtop[i])
                feat1_lbot[i] = self.att1(feat1_lbot[i])
                feat1_rbot[i] = self.att1(feat1_rbot[i])
        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats = self.sam12(res1_top[0])
        x2bot_samfeats = self.sam12(res1_bot[0])

        ## Output image at Stage 1
        stage1_img = torch.cat([x2top_samfeats, x2bot_samfeats], 2)
        stage1_img = self.conv1(stage1_img)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat)
        feat2_bot = self.stage2_encoder(x2bot_cat)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats = self.sam23(res2[0])
        stage2_img = self.conv2(x3_samfeats)
        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3 = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat)
        stage3_img = self.tail(x3_cat)
        out = self.last_conv(stage3_img)
        return [out,stage2_img,stage1_img]

if __name__ == '__main__':
    x = torch.randn(8, 3, 512, 512)
    model = TSISSFNet()
    res = model(x)
    print(res[0].shape)