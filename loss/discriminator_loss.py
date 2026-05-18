import torch
import torch.nn.functional as F

def discriminator_loss(real_pred, fake_pred):

    mean_real = torch.mean(real_pred)
    mean_fake = torch.mean(fake_pred)

    DRA_real = torch.sigmoid(real_pred - mean_fake) # DRA（xr, xf）
    DRA_fake = torch.sigmoid(fake_pred - mean_real) # DRA（xf, xr）

    loss_D = -torch.mean(torch.log(DRA_real + 1e-8)) - torch.mean(torch.log(1 - DRA_fake + 1e-8))
    loss_G = -torch.mean(torch.log(1 - DRA_real + 1e-8)) - torch.mean(torch.log(DRA_fake + 1e-8))
    return loss_D, loss_G