import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from natsort import natsorted
from glob import glob
from tqdm import tqdm

def calculate_metrics(clean_image_path, denoised_image_path):
    clean_images = natsorted(glob(os.path.join(clean_image_path, '*.png')))
    denoised_images = natsorted(glob(os.path.join(denoised_image_path, '*.png')))
    psnr_scores = []
    ssim_scores = []

    for clean_image, denoised_image in tqdm(zip(clean_images, denoised_images)):
        clean_img = cv2.imread(clean_image)
        denoised_img = cv2.imread(denoised_image)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

        psnr_scores.append(peak_signal_noise_ratio(clean_img, denoised_img, data_range=255))
        ssim_scores.append(structural_similarity(clean_img, denoised_img, 
                                                  channel_axis=2,
                                                  multichannel=True, 
                                                  data_range=255, 
                                                  win_size=7))  # Set win_size to a smaller odd value
    print(f"Average PSNR: {sum(psnr_scores) / len(psnr_scores)}")
    print(f"Average SSIM: {sum(ssim_scores) / len(ssim_scores)}")

if __name__ == "__main__":
    clean_image_path = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\original_patches"
    denoised_image_path = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\denoised_50"
    calculate_metrics(clean_image_path, denoised_image_path)
