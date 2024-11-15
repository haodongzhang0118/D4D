import argparse
from utils.GeneratePatches import generate_patches
from utils.CombineTwoFolders import CombineTwoFolders

"""
The training data we use is the combination of the training and validation data from the DIV2K dataset.
We first combine the training and validation data into a single folder, then generate patches from the combined data.

DIV2K dataset can be downloaded from kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")

print("Path to dataset files:", path)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument('--train_dir', type=str, help='Directory for train images')
    parser.add_argument('--val_dir', type=str, help='Directory for valid images')
    parser.add_argument('--tar_dir', type=str, help='Directory for combined images')
    parser.add_argument('--patch_dir', type=str, help='Directory for patched images')
    parser.add_argument('--ps', default=256, type=int, help='Patch size (e.g., 256 for 256x256 patches)')
    parser.add_argument('--num_cores', default=4, type=int, help='Number of CPU cores for parallel processing')

    args = parser.parse_args()
    train_dir = args.train_dir
    val_dir = args.val_dir
    tar_dir = args.tar_dir
    patch_dir = args.patch_dir
    patch_size = args.ps
    num_cores = args.num_cores

    print(f"Combining train_dir and val_dir to tar_dir")
    CombineTwoFolders(train_dir, val_dir, tar_dir)
    print(f"Generating patches from tar_dir to patch_dir")
    generate_patches(tar_dir, patch_dir, patch_size, num_cores)
    print("Preprocessing complete.")
