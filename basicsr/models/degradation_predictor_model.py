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

        pred_norm = torch.stack(
            [
                pred[:, 0] / 1.5,
                pred[:, 1] / 10.0,
                (pred[:, 2] - 20.0) / 80.0,
            ],
            dim=1,
        )

        target_norm = torch.stack(
            [
                self.degradation[:, 0] / 1.5,
                self.degradation[:, 1] / 10.0,
                (self.degradation[:, 2] - 20.0) / 80.0,
            ],
            dim=1,
        )

        loss = self.cri_deg(pred_norm, target_norm)

        loss.backward()
        self.optimizer_d.step()

        mae = torch.mean(torch.abs(pred_norm.detach() - target_norm), dim=0)

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

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.net_d.eval()

        total_mae = torch.zeros(3, device=self.device)
        total_samples = 0

        with torch.no_grad():
            for data in dataloader:
                lq = data["lq"].to(self.device)
                target = data["degradation"].to(self.device)

                pred = self.net_d(lq)

                batch_size = lq.size(0)

                total_mae += torch.abs(pred - target).mean(dim=0) * batch_size
                total_samples += batch_size

        total_mae /= total_samples

        self.net_d.train()

        self.log_dict = {
            "val_mae_blur": total_mae[0],
            "val_mae_noise": total_mae[1],
            "val_mae_jpeg": total_mae[2],
        }

        print(
            f"[Validation] Iteration {current_iter}: "
            f"Blur MAE: {total_mae[0].item():.6f}, "
            f"Noise MAE: {total_mae[1].item():.6f}, "
            f"JPEG MAE: {total_mae[2].item():.6f}"
        )
