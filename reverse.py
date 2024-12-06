import torch
from PIL import Image
from torchvision import transforms
from DDPMBackword import reverse_diffusion_from_noise
from model import NoiseEstimationClip, NoiseEstimationCLIP_pretrained

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def reverse(noise_image_path, checkpoint_path, diffusion_model_name, img_size):
    estimator = NoiseEstimationCLIP_pretrained()
    estimator = load_checkpoint(estimator, checkpoint_path)
    estimator.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator.to(device)

    transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),  # Converts to [0, 1]
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
            ])
    noise_image = Image.open(noise_image_path).convert('RGB')
    noise_image = transform(noise_image).unsqueeze(0).to(device)
    timestep = estimator(noise_image)
    print("Estimated timestep: ", timestep)

    denoised_image = reverse_diffusion_from_noise(noise_image_path, timestep, diffusion_model_name)
    return denoised_image