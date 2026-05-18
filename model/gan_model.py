import torch
import torch_npu
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils as nn_utils
from torch.optim import Adam
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from model.discriminator import Discriminator
from model.generator import Generator
from loss.Loss import Loss
from model.stage2_head import Stage2Head

import torch.nn.functional as F

class GANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('npu' if torch.npu.is_available() else 'cpu')
        
        # model build
        self.net_g = Generator().to(self.device)
        self.net_d = Discriminator().to(self.device)
        self.net_h = Stage2Head().to(self.device)

        # whether to enable dynamic learning rate
        lr_g = 1e-4
        lr_d = 2e-4
        lr_h = 1e-4
        self.rescale_lr = False

        # loss record
        self.loss_g = 0
        self.loss_d = 0
        self.loss_cls = 0
        self.loss_reg = 0
        self.loss_h =0

        # optimizer
        self.setup_optimizers(lr_g, lr_d, lr_h)

        # loss func
        self.loss_fn = Loss()

        self.WARMUP_EPOCH = 20

    def setup_optimizers(self, lr_g, lr_d, lr_h):
        beta1_g = 0.9  # exponential decay rate for the first moment estimates
        beta1_d = 0.9  # exponential decay rate for the first moment estimates

        # generator optimizer
        self.optimizer_g = Adam(
            self.net_g.parameters(),
            lr=lr_g,
            betas=(beta1_g, 0.999)
        )
        # discriminator optimizer
        self.optimizer_d = Adam(
            self.net_d.parameters(),
            lr=lr_d,
            betas=(beta1_d, 0.999)
        )
        # detector optimizer
        self.optimizer_h = Adam(
            self.net_h.parameters(),
            lr=lr_h,
            betas=(beta1_d, 0.999)
        )
 
        # for gan training, halve the learning rate every 10k steps
        if self.rescale_lr: 
            self.scheduler_g = StepLR(self.optimizer_g, step_size=10000, gamma=0.5)
            self.scheduler_d = StepLR(self.optimizer_d, step_size=10000, gamma=0.5) 
        
        self.optimizers = [self.optimizer_g, self.optimizer_d, self.optimizer_h]

    # data flow:
    # input data (lq, gt) → feed_data() load to device → optimize_parameters() train → output loss and update parameters
    def feed_data(self, data):
        """load and input a batch of data"""
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

        self.label = data['label'].to(self.device)
        self.w = data['w'].to(self.device)
        self.h = data['h'].to(self.device)
        
    def optimize_parameters(self,epoch):
        if epoch + 1 <= self.WARMUP_EPOCH:
            w_det = 0
        else :
            w_det = 0.003
        # ---------------------- 1. D training ----------------------
        self.net_d.train()
        for p in self.net_d.parameters():
            p.requires_grad_(True)

        # g is only used to generate fake images, no gradient needed
        with torch.no_grad():
            fake_img = self.net_g(self.lq)

        real_pred = self.net_d(self.gt)
        fake_pred = self.net_d(fake_img.detach())

        self.optimizer_d.zero_grad()
        l_d_total = self.loss_fn.get_d_loss(real_pred, fake_pred)
        l_d_total.backward()
        self.optimizer_d.step()
        # ---------------------- 2. G H training----------------------
        self.net_g.train()
        self.net_h.train()
        for p in self.net_d.parameters():
            p.requires_grad_(False)
        self.optimizer_g.zero_grad()
        self.optimizer_h.zero_grad()

        # G forward
        self.output = self.net_g(self.lq)
        fake_pred = self.net_d(self.output)
        real_pred = self.net_d(self.gt)  # d's parameters have requires_grad=false, will not be updated

        l_g = self.loss_fn.get_g_loss(self.gt, self.output, real_pred, fake_pred)
        cls_logits, box_refine = self.net_h(self.output)
        cls_loss = F.cross_entropy(cls_logits, self.label)
        target = torch.stack([(128 - self.w)/128.0, (128 - self.h)/128.0], dim=1)  
        reg_loss = F.smooth_l1_loss(box_refine, target)

        l_det = cls_loss + 100.0 * reg_loss 

        l_total = l_g + w_det * l_det  

        l_total.backward()
        self.optimizer_g.step()
        self.optimizer_h.step()

        # ---------------------- 3. logging & lr adjustment ----------------------
        self.loss_g = l_g.detach()
        self.loss_d = l_d_total.detach()
        self.loss_cls = cls_loss.detach()
        self.loss_reg = reg_loss.detach()
        self.loss_h = l_det.detach()

        if self.rescale_lr:
            self.scheduler_g.step()
            self.scheduler_d.step()

    def save(self, save_path):
        """save model weights"""
        state_dict = {
            'net_g': self.net_g.state_dict(),
            'net_d': self.net_d.state_dict(),
            'net_h': self.net_h.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_h':self.optimizer_h.state_dict()
        }
        if hasattr(self, 'net_g_ema'):
            state_dict['net_g_ema'] = self.net_g_ema.state_dict()
        torch.save(state_dict, save_path)

    def load_pretrain(self, load_path):
        """load model weights"""
        state_dict = torch.load(load_path, map_location=self.device)
        self.net_g.load_state_dict(state_dict['net_g'])
        self.net_d.load_state_dict(state_dict['net_d'])
        self.optimizer_g.load_state_dict(state_dict['optimizer_g'])
        self.optimizer_d.load_state_dict(state_dict['optimizer_d'])
        if 'net_g_ema' in state_dict and hasattr(self, 'net_g_ema'):
            self.net_g_ema.load_state_dict(state_dict['net_g_ema'])

    def load(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        self.net_g.load_state_dict(state_dict['net_g'])
        self.net_d.load_state_dict(state_dict['net_d'])
        self.net_h.load_state_dict(state_dict['net_h'])
        self.optimizer_g.load_state_dict(state_dict['optimizer_g'])
        self.optimizer_d.load_state_dict(state_dict['optimizer_d'])
        self.optimizer_h.load_state_dict(state_dict['optimizer_h'])
        if 'net_g_ema' in state_dict and hasattr(self, 'net_g_ema'):
            self.net_g_ema.load_state_dict(state_dict['net_g_ema'])
    



    