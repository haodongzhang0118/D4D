import torch
from torchvision import transforms
from PIL import Image
from diffusers import DDPMPipeline
from tqdm.auto import tqdm

def load_pretrained_model(model_name="google/ddpm-ema-cat-256"):
    """Load a pretrained diffusion model"""
    model_id = model_name
    pipeline = DDPMPipeline.from_pretrained(model_id)
    return pipeline.unet, pipeline.scheduler

def reverse_diffusion_from_noise(noised_image_path, start_timestep, model_name="google/ddpm-ema-cat-256"):
    """
    Perform reverse diffusion starting from a provided noised image at a specific timestep

    Args:
        noised_image_path: Path to the noised input image
        start_timestep: The timestep number of the noised image (e.g., 50)
    """
    model, scheduler = load_pretrained_model(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    noised_image = Image.open(noised_image_path).convert('RGB')
    x_t = transform(noised_image).unsqueeze(0).to(device)

    scheduler.set_timesteps(1000)
    timesteps = scheduler.timesteps[scheduler.timesteps <= start_timestep]

    # Reverse diffusion process
    with torch.no_grad():
        for t in tqdm(timesteps):
            noise_pred = model(x_t, t).sample
            scheduler_output = scheduler.step(
                noise_pred,
                t,
                x_t
            )
            x_t = scheduler_output.prev_sample

    final_image = tensor_to_image(x_t)

    return final_image

def tensor_to_image(tensor):
    """Convert a tensor to PIL Image"""
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).type(torch.uint8)
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(tensor)
