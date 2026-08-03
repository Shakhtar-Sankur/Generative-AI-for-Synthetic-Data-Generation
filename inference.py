import torch
import argparse
import os
from ddpm import DDPM, get_noise_schedule
from torchvision.utils import save_image

def generate_images(args):
    # getattr rather than args.device: callers construct their own Namespace
    # (the tests do), and a new flag should not break them.
    requested = getattr(args, 'device', 'auto')
    device = torch.device(requested if requested != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = DDPM().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    betas, alphas, alphas_cumprod = get_noise_schedule()
    betas, alphas, alphas_cumprod = betas.to(device), alphas.to(device), alphas_cumprod.to(device)
    images = torch.randn(args.num_images, 1, 64, 64, device=device)
    with torch.no_grad():
        for t in reversed(range(model.timesteps)):
            t_tensor = torch.full((args.num_images,), t, dtype=torch.long, device=device)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            noise_pred = model(images, t_tensor)
            images = (1 / torch.sqrt(alpha_t)) * (images - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred)
            if t > 0:
                images += torch.sqrt(beta_t) * torch.randn_like(images)
    os.makedirs(args.output, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, f"{args.output}/synthetic_{i}.png", normalize=True)
    print(f"Generated {args.num_images} images in {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/epoch_49.pth")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()
    generate_images(args)