import os
import cv2
import torch
from natsort import natsorted
from glob import glob
from PIL import Image
from torchvision import transforms
from DDPMBackword import reverse_diffusion_from_noise, load_pretrained_model
from model import NoiseEstimationClip, NoiseEstimationCLIP_pretrained
from utils.GeneratePatches import reconstruct_image
from tqdm.auto import tqdm

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def reverse(src_dir, tar_dir, clean_dir, checkpoint_path, diffusion_model_name, img_size, specific_timesteps):
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
    clean_files = natsorted(glob(os.path.join(clean_dir, '*.png')))
    index = 0
    for clean in tqdm(clean_files):
        clean_name = os.path.basename(clean).split('.')[0]
        lis_imgs = []
        lis_step = []
        print(f"The image's name is {clean_name}")
        while True: #for file in tqdm(files):
            if index >= len(files):
                break
            file = files[index]
            file_name = os.path.basename(file).split('.')[0][:-2]
            if clean_name == file_name:
                noise_image = Image.open(file).convert('RGB')
                noise_image = transform(noise_image).unsqueeze(0).to(device)
                t = torch.arange(specific_timesteps)
                logits = estimator(noise_image, t)
                timestep = (torch.argmax((100 * logits).softmax(dim=-1))).item()
                lis_step.append(timestep)
                lis_imgs.append(file)
            if clean_name != file_name:
                break
            index += 1
        timestep = sum(lis_step) // len(lis_step) + 1
        print(f"Average timestep: {timestep}")
        print(f"Now denoising image {clean_name}'s patches")
        for i in range(len(lis_imgs)):
            denoised_image = reverse_diffusion_from_noise(lis_imgs[i], timestep * 2, diffusion_model, scheduler)
            file_name_current = clean_name + f"_{i}.png"
            denoised_image.save(os.path.join(tar_dir, file_name_current))

def reverse2Whole(src_dir, clean_dir, tar_dir, img_size):
    clean_files = natsorted(glob(os.path.join(clean_dir, '*.png')))
    src_files = natsorted(glob(os.path.join(src_dir, '*.png')))
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    index = 0
    for i in tqdm(clean_files):
        file_name = os.path.basename(i)
        file_name = file_name.split('.')[0]
        clean_image = cv2.imread(i)
        h, w = clean_image.shape[:2]
        files = []
        while True:
            if index >= len(src_files):
                break
            j = src_files[index]
            src_name = os.path.basename(j)
            src_name = src_name.split('.')[0][:-2]
            if file_name == src_name:
                files.append(j)
            else:
                break
        index += 1
        denoised_image = reconstruct_image(files, h, w, img_size)
        file_name += ".png"
        cv2.imwrite(os.path.join(tar_dir, file_name), denoised_image)
