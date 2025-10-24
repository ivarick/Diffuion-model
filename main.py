import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm 
from model import UNet
from datamanager import Data

# diffusion process util
def get_beta_schedule(timesteps=1000, beta_start=0.001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }

def q_sample(x_start, t, noise, schedule):
    sqrt_alphas_cumprod_t = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def train(model, dataloader, optimizer, schedule, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader):
            images = batch.to(device)
            t = torch.randint(0, len(schedule['betas']), (images.size(0),)).to(device)
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, t, noise, schedule)
            predicted_noise = model(noisy_images, t)
            loss = nn.MSELoss()(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        
        if (epoch + 1) % 100 == 0:
            print(f"Generating samples at epoch {epoch + 1}...")
            generated_images = sample(model, schedule, device, num_images=5)
            for i, img in enumerate(generated_images):
                save_image(img.float() / 255.0, f"generated_epoch_{epoch+1}_{i}.png")

@torch.no_grad()
def sample(model, schedule, device, img_size=128, num_images=1, timesteps=1000):
    model.eval()
    x = torch.randn(num_images, 3, img_size, img_size).to(device)
    for i in tqdm(reversed(range(timesteps)), total=timesteps):
        t = torch.full((num_images,), i, dtype=torch.long, device=device)
        predicted_noise = model(x, t)
        beta_t = schedule['betas'][t].view(-1, 1, 1, 1)
        alpha_t = 1.0 - beta_t
        alphas_cumprod_t = schedule['alphas_cumprod'][t].view(-1, 1, 1, 1)
        alphas_cumprod_prev_t = schedule['alphas_cumprod_prev'][t].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t))
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alphas_cumprod_t)) * predicted_noise)
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean
    x = (x.clamp(-1, 1) + 1) / 2 
    x = (x * 255).type(torch.uint8)
    model.train() 
    return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16 
    epochs = 1000 # the model needs an huge amount of epochs
    timesteps = 1000
    root_dir = "data"
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])

    dataset = Data(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    schedule = get_beta_schedule(timesteps=timesteps)
    for k in schedule:
        schedule[k] = schedule[k].to(device)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    train(model, dataloader, optimizer, schedule, device, epochs=epochs)

    generated_images = sample(model, schedule, device, num_images=5)
    for i, img in enumerate(generated_images):
        save_image(img.float() / 255.0, f"generated_final_{i}.png")