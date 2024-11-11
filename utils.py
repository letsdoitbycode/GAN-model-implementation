import os
import torch
import torchvision
import torch.nn as nn
from torchvision.utils import save_image

def save_generated_images(generator, noise, epoch, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).detach().cpu()
    save_image(fake_images, os.path.join(output_dir, f"epoch_{epoch}.png"), normalize=True)
    generator.train()

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
