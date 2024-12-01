import numpy as np
from scipy import stats
import cv2

def load_image(image_path):
    """
    Load image from path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def normalize_with_clean_stats(clean_img, noisy_img):
    """
    Normalize both images using clean image statistics
    
    Args:
        clean_img: Clean image array
        noisy_img: Noisy image array
    
    Returns:
        tuple: (normalized_clean, normalized_noisy)
    """
    # Calculate mean and std of clean image
    clean_mean = np.mean(clean_img)
    clean_std = np.std(clean_img)
    
    print(f"Clean image - Mean: {clean_mean:.4f}, Std: {clean_std:.4f}")
    
    # Normalize both images using clean image statistics
    norm_clean = (clean_img - clean_mean) / clean_std
    norm_noisy = (noisy_img - clean_mean) / clean_std
    return norm_clean, norm_noisy

def reshape_images(clean_img, noisy_img, target_size=(256, 256)):
    """
    Reshape both images to target size
    """
    if clean_img.shape[:2] != target_size or noisy_img.shape[:2] != target_size:
        if len(clean_img.shape) == 3:
            reshaped_clean = cv2.resize(clean_img, target_size[::-1])
            reshaped_noisy = cv2.resize(noisy_img, target_size[::-1])
        else:
            reshaped_clean = cv2.resize(clean_img, target_size[::-1], interpolation=cv2.INTER_LINEAR)
            reshaped_noisy = cv2.resize(noisy_img, target_size[::-1], interpolation=cv2.INTER_LINEAR)
        print(f"Images reshaped from {clean_img.shape[:2]} to {target_size}")
    else:
        reshaped_clean = clean_img
        reshaped_noisy = noisy_img
        print("Images already at target size")
    
    return reshaped_clean, reshaped_noisy

def estimate_noise_level_mse(clean_img, noisy_img):
    """
    Estimate noise level using MSE method
    """
    noise = noisy_img - clean_img
    sigma_est = np.std(noise)
    mse = np.mean(noise**2)
    
    print(f"MSE: {mse:.6f}")
    print(f"Estimated sigma: {sigma_est:.6f}")
    
    return sigma_est

def estimate_noise_level_mad(clean_img, noisy_img):
    """
    Estimate noise level using MAD method
    """
    noise = noisy_img - clean_img
    mad = stats.median_abs_deviation(noise.flatten())
    sigma_est = mad * 1.4826
    
    print(f"MAD: {mad:.6f}")
    print(f"Estimated sigma: {sigma_est:.6f}")
    
    return sigma_est

def estimate_noise_level_snr(clean_img, noisy_img):
    """
    Estimate noise level using SNR method
    """
    signal_power = np.mean(clean_img**2)
    noise = noisy_img - clean_img
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    
    print(f"Signal Power: {signal_power:.6f}")
    print(f"Noise Power: {noise_power:.6f}")
    print(f"SNR (dB): {snr:.6f}")
    
    return snr

def main(clean_path, noisy_path):
    """
    Main function to estimate noise from image paths
    """
    try:
        # Load images
        print("Loading images...")
        clean_img = load_image(clean_path)
        noisy_img = load_image(noisy_path)
        
        # Reshape first
        print("\nReshaping images...")
        clean_img, noisy_img = reshape_images(clean_img, noisy_img)
        
        # Normalize using clean image statistics
        print("\nNormalizing images using clean image statistics...")
        norm_clean, norm_noisy = normalize_with_clean_stats(clean_img, noisy_img)
        
        # Estimate noise using different methods
        print("\nMSE Method:")
        sigma_mse = estimate_noise_level_mse(norm_clean, norm_noisy)
        
        print("\nMAD Method:")
        sigma_mad = estimate_noise_level_mad(norm_clean, norm_noisy)
        
        print("\nSNR Method:")
        snr = estimate_noise_level_snr(norm_clean, norm_noisy)
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Usage example:
"""
clean_path = "path/to/your/clean/image.png"
noisy_path = "path/to/your/noisy/image.png"
main(clean_path, noisy_path)
"""
clean_path = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\original\0010.png"
noisy_path = r"C:\Users\haodo\OneDrive\Desktop\CS 2470\Final Project\Denoise Dataset\CBSD68\noisy50\0010.png"
main(clean_path, noisy_path)