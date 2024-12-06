import cv2
import os
import torch
from natsort import natsorted
from glob import glob
from PIL import Image
from torchvision import transforms
from DDPMBackword import reverse_diffusion_from_noise, load_pretrained_model
from model import NoiseEstimationClip, NoiseEstimationCLIP_pretrained
from tqdm.auto import tqdm

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def reverse(src_dir, tar_dir, checkpoint_path, diffusion_model_name, img_size, specific_timesteps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    estimator = NoiseEstimationCLIP_pretrained()
    estimator = load_checkpoint(estimator, checkpoint_path)
    estimator.eval()
    estimator.to(device)

    diffusion_model, scheduler = load_pretrained_model(diffusion_model_name)
    diffusion_model.eval()
    diffusion_model.to(device)

    transform = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),  # Converts to [0, 1]
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
            ])
    
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    files = natsorted(glob(os.path.join(src_dir, '*.png')))
    for file in tqdm(files):
        file_name = os.path.basename(file).split('.')[0]
        noise_image = Image.open(file).convert('RGB')
        noise_image = transform(noise_image).unsqueeze(0).to(device)
        t = torch.arange(specific_timesteps)
        timestep = estimator(noise_image, t)
        print("Estimated timestep: ", timestep)

        denoised_image = reverse_diffusion_from_noise(file, timestep, diffusion_model, scheduler)
        cv2.imwrite(os.path.join(tar_dir, file_name), denoised_image)