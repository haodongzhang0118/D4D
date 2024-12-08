import os
import cv2
import numpy as np
from natsort import natsorted
from glob import glob
from tqdm import tqdm

def add_noise(src_dir, tar_dir, noise_level=25):
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    files = natsorted(glob(os.path.join(src_dir, '*.png')))
    for file in tqdm(files):
        img = cv2.imread(file)
        noise = np.random.normal(0, noise_level, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(tar_dir, os.path.basename(file)), noisy_img)

if __name__ == "__main__":
    src_dir = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\original_patches"
    tar_dir = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\noisy\50"
    noise_level = 50
    add_noise(src_dir, tar_dir, noise_level)