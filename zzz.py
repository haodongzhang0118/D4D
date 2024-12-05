import torch
import os
from torchvision.utils import save_image
from pathlib import Path
from torchvision import transforms
from PIL import Image

def save_tensor_images(tensor, output_folder):
    """
    Save each image from a batch tensor to individual PNG files.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (B, C, H, W)
        output_folder (str): Path to the output folder
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Ensure the tensor is in the correct range [0, 1]
    # If your tensor is in range [-1, 1], uncomment the next line
    # tensor = (tensor + 1) / 2
    
    # For each image in the batch
    for i in range(tensor.shape[0]):
        # Extract single image tensor (C, H, W)
        img_tensor = tensor[i]
        
        # Create output path
        output_path = os.path.join(output_folder, f'img_{i}.png')
        
        # Save the image
        save_image(img_tensor, output_path)
        
        # Optional: Print progress every 50 images
        if (i + 1) % 50 == 0:
            print(f'Saved {i + 1} images')

# Example usage
if __name__ == "__main__":
    # Example tensor with shape (256, 3, 256, 256)
    images_tensor = torch.load(r"C:\Users\haodo\OneDrive\Desktop\img.pt")  # Replace with your actual tensor
    
    # Specify output folder
    output_folder = r"C:\Users\haodo\OneDrive\Desktop\img"
    
    # Save all images
    save_tensor_images(images_tensor, output_folder)
    print(f'All images saved to {output_folder}')