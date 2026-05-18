import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from loss.discriminator_loss import discriminator_loss


class Loss(nn.Module):
    def __init__(
        self,
        w_mse=10.0,      
        w_gan=0.01, 
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_gan = w_gan

    def get_mse_loss(self, gt, sr):
        return nn.MSELoss()(gt, sr)
    
    def get_d_loss(self, real_pred, fake_pred):
        loss_d, _ = discriminator_loss(real_pred, fake_pred)
        return loss_d

    def get_g_loss(self, gt, sr, real_pred, fake_pred):
        # ----- 2. pixel loss -----
        loss_mse = self.get_mse_loss(gt, sr)
        #print(f"loss_pix:{loss_mse}")

        # ----- 3. GAN loss (RaGAN) -----
        _, loss_gan = discriminator_loss(real_pred, fake_pred)
        #print(f"loss_gan:{loss_gan}")

        # ----- 4. total loss -----
        total = (
            self.w_mse * loss_mse +
            self.w_gan * loss_gan
        )

        return total
