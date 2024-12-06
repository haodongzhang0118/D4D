import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from natsort import natsorted
from glob import glob
from tqdm import tqdm

# parser = argparse.ArgumentParser(description='Generate patches from images')
# parser.add_argument('--src_dir', default='path/to/original/images', type=str, help='Directory for original images')
# parser.add_argument('--tar_dir', default='path/to/save/patches', type=str, help='Directory for image patches')
# parser.add_argument('--ps', default=256, type=int, help='Patch size (e.g., 256 for 256x256 patches)')
# parser.add_argument('--num_cores', default=4, type=int, help='Number of CPU cores for parallel processing')

# args = parser.parse_args()

# src_dir = args.src_dir
# tar_dir = args.tar_dir
# patch_size = args.ps
# num_cores = args.num_cores


def process_image(file_path, patch_size, tar_dir, mean=0, std=0):
    img = cv2.imread(file_path)
    h, w = img.shape[:2]
    file_name = os.path.basename(file_path).split('.')[0]
    patch_id = 0

    pad_height = (patch_size - h % patch_size) % patch_size
    pad_width = (patch_size - w % patch_size) % patch_size

    #padded_img = cv2.copyMakeBorder(img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if pad_height > 0 or pad_width > 0:
        normal_padding = np.random.normal(loc=mean, scale=std, size=(h + pad_height, w + pad_width, 3)) / 255.0
        normal_padding[:h, :w] = img / 255.0
        padded_img = normal_padding * 255.0
    else:
        padded_img = img

    padded_h, padded_w = padded_img.shape[:2] 

    # Generate patches
    for start_row in range(0, padded_h, patch_size):
        for start_col in range(0, padded_w, patch_size):
            end_row = start_row + patch_size
            end_col = start_col + patch_size
            patch = padded_img[start_row:end_row, start_col:end_col]
            patch_name = f"{file_name}_{patch_id}.png"
            cv2.imwrite(os.path.join(tar_dir, patch_name), patch)
            patch_id += 1

def generate_patches(src_dir, tar_dir, patch_size, num_cores, mean=0, std=0):
    print('std: ', std)
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    files = natsorted(glob(os.path.join(src_dir, '*.png')))
    Parallel(n_jobs=num_cores)(delayed(process_image)(file, patch_size, tar_dir, mean, std) for file in tqdm(files))
    print("Patching complete.")

def reconstruct_image(patch_files, original_h, original_w, patch_size):
    padded_h = ((original_h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((original_w + patch_size - 1) // patch_size) * patch_size

    reconstructed_img = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)

    for patch_file in patch_files:
        patch = cv2.imread(patch_file)
        base_name = os.path.basename(patch_file)
        index_info = base_name.split('.')[0].split('_')[-1]
        patch_id = int(index_info)

        row_index = patch_id // (padded_w // patch_size)
        col_index = patch_id % (padded_w // patch_size)

        start_row = row_index * patch_size
        start_col = col_index * patch_size

        reconstructed_img[start_row:start_row + patch_size, start_col:start_col + patch_size] = patch

    final_img = reconstructed_img[:original_h, :original_w]

    return final_img

