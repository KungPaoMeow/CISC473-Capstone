import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DegradationPredictorModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.net_d = build_network(opt["network_d"])
        self.net_d = self.model_to_device(self.net_d)

        self.print_network(self.net_d)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_d.train()

        train_opt = self.opt["train"]

        self.cri_deg = torch.nn.SmoothL1Loss()

        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(),
            lr=train_opt["optim_d"]["lr"],
            betas=tuple(train_opt["optim_d"].get("betas", (0.9, 0.99))),
        )

        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        self.degradation = data["degradation"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_d.zero_grad()

        pred = self.net_d(self.lq)

        loss = self.cri_deg(pred, self.degradation)

        loss.backward()
        self.optimizer_d.step()

        mae = torch.mean(torch.abs(pred.detach() - self.degradation), dim=0)

        self.log_dict = {
            "l_deg": loss.detach(),
            "mae_blur": mae[0],
            "mae_noise": mae[1],
            "mae_jpeg": mae[2],
        }

    def get_current_visuals(self):
        return {}

    def save(self, epoch, current_iter):
        self.save_network(self.net_d, "net_d", current_iter, param_key="params")
        self.save_training_state(epoch, current_iter)
