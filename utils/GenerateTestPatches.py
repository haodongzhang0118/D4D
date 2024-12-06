import argparse
from utils.GeneratePatches import generate_patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument('--tar_dir', type=str, help='Directory for combined images')
    parser.add_argument('--patch_dir', type=str, help='Directory for patched images')
    parser.add_argument('--noise_level', default=0, type=int, help='Noise level')
    parser.add_argument('--ps', default=256, type=int, help='Patch size (e.g., 256 for 256x256 patches)')
    parser.add_argument('--num_cores', default=6, type=int, help='Number of CPU cores for parallel processing')
    
    args = parser.parse_args()
    tar_dir = args.tar_dir
    patch_dir = args.patch_dir
    patch_size = args.ps
    num_cores = args.num_cores
    noise_level = args.noise_level

    print('Noise level: ', noise_level)
    generate_patches(tar_dir, patch_dir, patch_size, num_cores, 0, noise_level)