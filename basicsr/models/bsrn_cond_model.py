from collections import OrderedDict

import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class BSRNCondModel(SRModel):
    def feed_data(self, data):

        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)

        degradation = data["degradation"].to(self.device)

        # "normalize" the degradation parameters so that they are roughly in the same range as each other (roughly 0 to 1)
        self.degradation = torch.stack(
            [
                torch.div(degradation[:, 0], 1.5),
                torch.div(degradation[:, 1], 10),
                torch.div((torch.sub(degradation[:, 2], 60)), 40),
                torch.div(degradation[:, 3], 4),
            ],
            1,
        )

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.degradation)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.optimizer_g.step()

        if current_iter % 100 == 0:
            for i, layer in enumerate(self.net_g.film):
                print(
                    f"Iteration {current_iter}, Block {i + 1}:",
                    "weight =",
                    layer.weight.abs().mean().item(),
                    "bias =",
                    layer.bias.abs().mean().item(),
                )

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.degradation)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.degradation)
            self.net_g.train()

    def init_training_settings(self):
        # # freeze generator parameters
        # self.net_g.requires_grad_(False)

        # # unfreeze conditioning components
        # for name, param in self.net_g.named_parameters():
        #     if name.startswith(
        #         ("degradation_encoder.", "film.", "c1.", "c2.", "upsampler.")
        #     ):
        #         param.requires_grad_(True)

        super().init_training_settings()

    def setup_optimizers(self):
        train_opt = self.opt["train"]

        conditioning_params = []
        reconstruction_params = []
        backbone_params = []

        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith(("degradation_encoder.", "film.")):
                conditioning_params.append(param)

            elif name.startswith(("c1.", "c2.", "upsampler.")):
                reconstruction_params.append(param)

            else:
                backbone_params.append(param)

        optim_params = [
            {"params": conditioning_params, "lr": 1e-4},
            {"params": reconstruction_params, "lr": 1e-5},
            {"params": backbone_params, "lr": 1e-6},
        ]

        optim_type = train_opt["optim_g"].pop("type")

        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )

        self.optimizers.append(self.optimizer_g)
