import torch
from basicsr.archs.BSRN_arch import BSRN
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn

# class ConditionalESDB(nn.Module):
#     def __init__(self, original_block, num_feat, embed_dim=32):
#         super().__init__()

#         self.block = original_block

#         # Generate channel-wise scale and bias
#         self.film = nn.Linear(embed_dim, num_feat * 2)

#         # Identity initialisation:
#         # gamma = 0, beta = 0
#         nn.init.zeros_(self.film.weight)
#         nn.init.zeros_(self.film.bias)

#     def forward(self, x, degradation_embedding):

#         # Original pretrained ESDB
#         features = self.block(x)

#         # Generate FiLM parameters
#         gamma, beta = self.film(degradation_embedding).chunk(2, dim=1)

#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#         beta = beta.unsqueeze(-1).unsqueeze(-1)

#         # Degradation-conditioned features
#         features = features * (1 + gamma) + beta

#         return features


@ARCH_REGISTRY.register()
class BSRNCond(BSRN):
    def __init__(
        self,
        num_in_ch=3,
        num_feat=64,
        num_block=8,
        num_out_ch=3,
        upscale=4,
        conv="BSConvU",
        upsampler="pixelshuffledirect",
        p=0.25,
        degradation_dim=4,
        embed_dim=32,
    ):

        assert num_block == 8, "Original BSRN uses 8 ESDB blocks"

        super().__init__(
            num_in_ch=num_in_ch,
            num_feat=num_feat,
            num_block=num_block,
            num_out_ch=num_out_ch,
            upscale=upscale,
            conv=conv,
            upsampler=upsampler,
            p=p,
        )

        self.degradation_encoder = nn.Sequential(
            nn.Linear(degradation_dim, embed_dim), nn.GELU()
        )

        # One FiLM projection per ESDB block
        self.film = nn.ModuleList(
            [nn.Linear(embed_dim, num_feat * 2) for _ in range(num_block)]
        )

        # Initially reproduce original BSRN behaviour
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, input, degradation=None):

        batch_size = input.shape[0]

        if degradation is None:
            degradation = input.new_zeros(
                batch_size, self.degradation_encoder[0].in_features
            )

        z = self.degradation_encoder(degradation)

        # Original BSRN feature extraction
        input = torch.cat([input, input, input, input], dim=1)

        out_fea = self.fea_conv(input)

        # Process the original ESDB blocks
        block_outputs = []
        x = out_fea

        for block, film in zip(
            [self.B1, self.B2, self.B3, self.B4, self.B5, self.B6, self.B7, self.B8],
            self.film,
        ):
            x = block(x)

            gamma, beta = film(z).chunk(2, dim=1)

            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]

            x = x * (1 + gamma) + beta

            block_outputs.append(x)

        # Original BSRN feature aggregation
        trunk = torch.cat(block_outputs, dim=1)

        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output
