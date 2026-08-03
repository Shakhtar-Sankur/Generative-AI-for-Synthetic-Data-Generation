import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from ddpm import DDPM, get_noise_schedule

def train(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageFolder(args.dataset, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = DDPM().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    betas, _, alphas_cumprod = get_noise_schedule()
    for epoch in range(args.epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.cuda()
            batch_size = images.size(0)
            t = torch.randint(0, model.timesteps, (batch_size,)).cuda()
            noise = torch.randn_like(images)
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
            noisy_images = sqrt_alphas_cumprod * images + sqrt_one_minus_alphas * noise
            optimizer.zero_grad()
            pred = model(noisy_images, t)
            loss = torch.mean((pred - images) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mednist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args)