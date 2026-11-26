import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY



class DegradationEncoder(nn.Module):
    """Extract a global degradation (frequency domain) representation from the LR image."""

    def __init__(self, num_in_ch, embed_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_ch * 3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(64, embed_dim),
        )

    def forward(self, x):
        # return self.encoder(x)

        # x: [B, C, H, W]

        # full complex Fourier spectrum
        fft = torch.fft.fft2(x, norm="ortho")

        # center zero frequency for a more natural spatial layout
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        magnitude = torch.abs(fft)

        # log magnitude compresses the dynamic range
        log_magnitude = torch.log1p(magnitude)

        # represent phase continuously
        eps = 1e-8
        phase_real = fft.real / (magnitude + eps)   # COSINE of phase(u,v)
        phase_imag = fft.imag / (magnitude + eps)   # SINE of phase(u,v)

        # phase is unreliable where magnitude is very small - suppress these locations
        threshold = 0.01 * magnitude.mean(dim=(-2, -1), keepdim=True)
        phase_weight = magnitude / (magnitude + threshold + eps)

        phase_real = phase_real * phase_weight
        phase_imag = phase_imag * phase_weight

        # [B, C*3, H, W]
        freq_input = torch.cat([log_magnitude, phase_real, phase_imag], dim=1)

        return self.encoder(freq_input)


class FrequencyCorrectionHead(nn.Module):
    """Predict an HR-resolution (spatial) correction conditioned on degradation."""

    def __init__(self, num_feat, num_out_ch, embed_dim=32):
        super().__init__()

        # generate channel-wise modulation parameters from degradation
        self.condition = nn.Linear(embed_dim, num_feat * 2)

        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_out = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # start with no correction, preserving the original RRDB output
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, feat, degradation):
        # degradation: [B, embed_dim]
        # gamma, beta: [B, num_feat, 1, 1]
        gamma, beta = self.condition(degradation).chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # FiLM-style conditioning.
        feat = feat * (1 + gamma) + beta

        feat = self.lrelu(self.conv1(feat))
        feat = self.lrelu(self.conv2(feat))

        return self.conv_out(feat)
    

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)  # [b x num_feat x h x w] -> [b x num_grow_ch x h x w]
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)   # [b x (num_feat + 4*num_grow_ch) x h x w] -> [b x num_feat x h x w]

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        # all [b x num_feat x h x w] -> [b x num_feat x h x w]
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNetCond(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale_factor=4, num_feat=18, num_block=3, num_grow_ch=9):
        super(RRDBNetCond, self).__init__()
        self.scale = scale_factor
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)   # [b x num_in_ch x h x w] -> [b x num_feat x h x w]
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch) # [b x num_feat x h x w] -> [b x num_feat x h x w]
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)     # [b x num_feat x h x w] -> [b x num_feat x h x w]
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # Degradation conditioning
        self.degradation_encoder = DegradationEncoder(num_in_ch, embed_dim=32)

        # Frequency correction branch
        self.frequency_head = FrequencyCorrectionHead(num_feat, num_out_ch, embed_dim=32)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        print(f"number of parameters in model: {self.calculate_num_params()}")

    def forward(self, x, iter=None):
        degradation = self.degradation_encoder(x)   # estimate degradation from the LR input

        #################################### original RRDB network ####################################
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=self.scale, mode='nearest')))
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)    # RGB output
        ###############################################################################################

        correction = self.frequency_head(feat, degradation)


        if self.training and iter and iter % 100 == 0:
            logger = get_root_logger()
            logger.info(
                f"RRDB std: {out.std().item():.5f}, "
                f"Correction std: {correction.std().item():.5f}, "
                f"Correction/Output ratio: "
                f"{correction.std().item() / (out.std().item() + 1e-8):.3f}"
            )

        return out + 0.2 * correction

    def calculate_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)