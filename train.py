import os
import random
import csv
import torch
import torch_npu
from torch.utils.data import DataLoader, random_split
import torch.nn.utils as nn_utils

from tqdm import tqdm
import numpy as np

from model.gan_model import GANModel
from datasets.mydatasets import MyDataset

import matplotlib.pyplot as plt

from utils.paths import get_dir, get_path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_gan_losses(log_path, save_path):
    """
    Plot the curves according to the gan_log.csv file written during the training process.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    epochs = [int(r["epoch"]) for r in rows]

    plt.figure(figsize=(12, 8))

    # G Loss 
    plt.plot(epochs, [float(r["g_loss"]) for r in rows], label="G Loss", color="red")

    # D Loss 
    plt.plot(epochs, [float(r["d_loss"]) for r in rows], label="D Loss", color="blue")

    # Val L1
    if rows and rows[0].get("val_l1"):
        plt.plot(epochs, [float(r["val_l1"]) for r in rows], label="Val L1", color="green")

    # Val PSNR
    if rows and rows[0].get("val_psnr"):
        plt.plot(epochs, [float(r["val_psnr"]) for r in rows], label="Val PSNR", color="purple")

    # Val SSIM
    if rows and rows[0].get("val_ssim"):
        plt.plot(epochs, [float(r["val_ssim"]) for r in rows], label="Val SSIM", color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Metric Value")
    plt.title("GAN Training Loss & Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"=> GAN Loss Curve Saved to: {save_path}")

# GAN train
def train():
    set_seed(42)

    # basic config
    device = torch.device('npu' if torch.npu.is_available() else 'cpu')
    batch_size = 8
    num_workers = 4
    epochs = 50
    
    # model create
    model = GANModel().to(device)
    # data load
    dataset = MyDataset(
        lq_root=get_dir("datasets.LR"),
        gt_root=get_dir("datasets.HR"),
        meta_root=get_dir("datasets.META"),
        hr_scale=4.0,
        recursive=False
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )

    # create log
    checkpoint_dir = get_dir("train.checkpoint_dir", create=True)
    log_path = get_path("train.log_csv")
    loss_curve_path = get_path("train.loss_curve")

    with open(log_path, "w") as f:
        f.write("epoch,g_loss,d_loss,cls_loss,reg_loss,h_loss,val_l1,val_psnr,val_ssim\n")

    global_step = 0

    # training loop
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_detector_loss = 0.0
        iter_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for data in pbar:

            model.feed_data(data)

            # optimize
            model.optimize_parameters(epoch)

            # record loss
            g_loss = model.loss_g.item()
            d_loss = model.loss_d.item()
            cls_loss = model.loss_cls.item()
            reg_loss = model.loss_reg.item()
            detector_loss = model.loss_h.item()

            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
            epoch_cls_loss += cls_loss
            epoch_reg_loss += reg_loss
            epoch_detector_loss += detector_loss

            iter_count += 1
            global_step += 1

            pbar.set_postfix({
                "G_loss": f"{g_loss:.4f}",
                "D_loss": f"{d_loss:.4f}",
                "H_loss":f"{detector_loss:.4f}",
            })

        # epoch mean
        avg_g = epoch_g_loss / iter_count
        avg_d = epoch_d_loss / iter_count
        avg_cls = epoch_cls_loss / iter_count
        avg_reg = epoch_reg_loss / iter_count
        avg_h = epoch_detector_loss /  iter_count

        print(f"[Epoch {epoch+1}] G={avg_g:.6f}, D={avg_d:.6f}, H+{avg_h:.6f}")

        # write to log (gan phase)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_g},{avg_d},{avg_cls},{avg_reg},{avg_h},,,\n")

        # validation (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            # save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"gan_epoch{epoch+1}.pth")
            model.save(ckpt_path)

    print("gan training completed!")
    print("plotting loss graph...")
    plot_gan_losses(log_path, loss_curve_path)
    print("done!")


if __name__ == "__main__":
    train()
