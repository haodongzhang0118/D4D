import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

class DDPMForward:
    def __init__(
        self,
        transform=None,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        img_size=256,
        device=None,
        img_type='clean'
    ):
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_type = img_type
        print(f"Using device: {self.device}")

        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps) # .to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def prepare_single_image(self, image_path):
        """Load and preprocess a single image to tensor"""
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),  # Converts to [0, 1]
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
            ])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)  # Returns (1, 3, 256, 256)

    def forward_diffusion(self, x_0, specific_timesteps):
        """
        Forward diffusion process for a single timestep

        Args:
            x_0: Initial image tensor
            t: Timestep
        Returns:
            Noised image at timestep t and the noise added
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_clip = self.sqrt_alphas_cumprod[:specific_timesteps]
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod_clip.view(1, -1, 1, 1, 1)

        sqrt_one_minus_alphas_cumprod_clip = self.sqrt_one_minus_alphas_cumprod[:specific_timesteps]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod_clip.view(1, -1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t

    def Load_all_images(
        self,
        image_dir=None,
        clean_image=None,
        num_cores=6
    ):
        """
        Perform complete forward diffusion process
        """
        assert (image_dir is not None) or (clean_image is not None) , "Either image_dir or clean_image must be provided"
        if clean_image is not None:
            x_0 = torch.load(clean_image)
        else:
            # Process images in parallel and collect results
            image_dir = Path(image_dir)
            image_paths = list(image_dir.glob("*.png"))
            processed_images = Parallel(n_jobs=num_cores)(
                delayed(self.prepare_single_image)(str(img_path))
                for img_path in tqdm(image_paths)
            )
            # Stack all processed images along batch dimension
            x_0 = torch.stack(processed_images, dim=0)  # Shape: (42744, 1, 3, 256, 256)
            save_name = self.img_type + "Image.pt"
            torch.save(x_0, save_name)
        return x_0

    def forward_diffusion_all_steps(
        self,
        x_0,
        specific_timesteps=256
    ):
        x_0 = x_0 # .to(self.device)
        x_t = self.forward_diffusion(x_0, specific_timesteps)
        return x_t

    @staticmethod
    def tensor_to_image(tensor):
        """Convert tensor back to PIL Image"""
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).type(torch.uint8)
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(tensor)

# Example usage
"""
if __name__ == "__main__":
    import os
    from einops import rearrange
    ddpm = DDPMForward(
        num_timesteps=1000,
        img_size=256
    )
    x_0 = ddpm.Load_all_images(
        r"path\to\images",
    )
    x_t = ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=256)
    x_t = rearrange(x_t, 'b t c h w -> t b c h w')
    # # # Ensure the directory exists before saving files
    save_path = r"path\to\save"
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
    # # Save results
    for i in range(2):
        for t, img in enumerate(x_t):
            ddpm.tensor_to_image(x_t[t, i, :, :, :]).save(os.path.join(save_path, f"image{i}_noised_t{t}.png"))  # Use os.path.join for path construction
"""