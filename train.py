import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from models.generator import Generator
from models.discriminator import Discriminator
from utils import save_generated_images, initialize_weights

# Hyperparameters
batch_size = 128
lr = 0.0002
num_epochs = 50
noise_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)
initialize_weights(generator)
initialize_weights(discriminator)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = generator(noise)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        optimizer_d.zero_grad()
        output_real = discriminator(real_images).squeeze()
        loss_d_real = criterion(output_real, real_labels)
        output_fake = discriminator(fake_images.detach()).squeeze()
        loss_d_fake = criterion(output_fake, fake_labels)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        output_fake = discriminator(fake_images).squeeze()
        loss_g = criterion(output_fake, real_labels)
        loss_g.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                  f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

    # Save generated images every few epochs
    if epoch % 5 == 0:
        save_generated_images(generator, noise, epoch)
