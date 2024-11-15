import os
import shutil
from tqdm import tqdm

def copy_pngs(source_folder, destination_folder):
    for file_name in tqdm(os.listdir(source_folder)):
        if file_name.endswith('.png'):
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(source_path, destination_path)

def CombineTwoFolders(folder_a, folder_b, folder_c):
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
    print(f"Copying PNGs from {folder_a} to {folder_c}")
    copy_pngs(folder_a, folder_c)
    print(f"Copying PNGs from {folder_b} to {folder_c}")
    copy_pngs(folder_b, folder_c)

