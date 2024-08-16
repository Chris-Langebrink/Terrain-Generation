import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, img_size, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, conditional_images,n , cfg_scale=3,dem_min =0,dem_max=1000):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 4, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditional_images)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # return x

        # Separate DEM and RGB channels
        dem_generated = x[:, 0, :, :].unsqueeze(1)
        satellite_generated = x[:, 1:4, :, :]

        # De-normalize DEM values to original range
        dem_generated = (dem_generated.clamp(-1,1) + 1) / 2 * (dem_max - dem_min) + dem_min

        # Process satellite images as before
        satellite_generated = (satellite_generated.clamp(-1, 1) + 1) / 2
        satellite_generated = (satellite_generated * 255).type(torch.uint8)

        return dem_generated, satellite_generated


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(args.image_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, conditioning_images) in enumerate(pbar):
            images = images.to(device)
            conditioning_images = conditioning_images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                conditioning_images = None
            predicted_noise = model(x_t, t, conditioning_images)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # if epoch % 2 == 0:
        # conditioning_images = conditioning_images.to(device)
        dem_images, satellite_images = diffusion.sample(model, conditioning_images ,n=images.shape[0])
        # ema_sampled_images = diffusion.sample(ema_model, n=len(conditioning_images), labels=conditioning_images)
        # plot_images(sampled_images)
        save_images(dem_images,satellite_images, os.path.join("results", args.run_name, f"{epoch}"))
        # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}_ckpt.pt"))
        # torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"{epoch}_optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Conditional_8"
    args.epochs = 100
    args.batch_size = 6
    args.image_size = 64
    args.num_samples = 100
    args.noise_steps = 1000

    args.start_samples = None
    args.end_samples = None

    args.classifier_path = r"C:\Users\USER\Desktop\Tiles\Classifier"
    args.DEM_path = r"C:\Users\USER\Desktop\Tiles\DEM"
    args.satellite_path = r"C:\Users\USER\Desktop\Tiles\Satellite"

    # args.classifier_path = "/scratch/lngchr014/data/Classifier_256"
    # args.DEM_path = "/scratch/lngchr014/data/DEM_256"
    # args.satellite_path = "/scratch/lngchr014/data/Satellite_256"

    args.device = "cuda"
    args.lr = 3e-4
    train(args)



if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./conditional_ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

    # Calculate total number of parameters
    # total_params = sum(p.numel() for p in model.parameters())

    # # Calculate trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f'Total parameters: {total_params}')
    # print(f'Trainable parameters: {trainable_params}')
    # 370 258 180

