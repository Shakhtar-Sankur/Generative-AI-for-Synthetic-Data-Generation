import os
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device if args.device != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = DDPM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    betas, _, alphas_cumprod = get_noise_schedule()
    # the schedule is indexed by t, which lives on the model's device
    alphas_cumprod = alphas_cumprod.to(device)
    for epoch in range(args.epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            t = torch.randint(0, model.timesteps, (batch_size,), device=device)
            noise = torch.randn_like(images)
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alphas = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
            noisy_images = sqrt_alphas_cumprod * images + sqrt_one_minus_alphas * noise
            optimizer.zero_grad()
            pred = model(noisy_images, t)
            # DDPM supervises the *noise*, which is what the reverse process in
            # inference.py subtracts. This used to compare against `images`,
            # training the network to output the clean image while sampling
            # treated its output as noise - the two halves disagreed, so the
            # sampler could only ever produce nonsense.
            loss = torch.mean((pred - noise) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(),
                   os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mednist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()
    train(args)