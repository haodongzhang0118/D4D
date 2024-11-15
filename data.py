from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.DDPMForward import DDPMForward, DDPMForward_array

class NoiseEstimationMixedDataset(Dataset):
    def __init__(self, image_dir, transform=None, clean_image=None, img_size=256, num_timesteps=1000, specific_timesteps=256, saved_all_data_first=False):
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.specified_timesteps = specific_timesteps
        self.transform = transform
        self.ddpm = DDPMForward_array(transform=self.transform, num_timesteps=num_timesteps, img_size=img_size)
        self.saveFirst = saved_all_data_first
        if self.saveFirst:
            # Load All image to torch tensor first to save Dataloader time. It will generate a pt file called cleanImage.pt which contains this tensor
            self.x_0 = self.ddpm.Load_all_images(image_dir=image_dir, clean_image=clean_image) # About 32 GB of data

    def __len__(self):
        return len(list(self.image_dir.glob("*.png")))
    
    def __getitem__(self, idx):
        if self.saveFirst:
            x_0 = self.x_0[idx]
            x_t = self.ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=self.specified_timesteps)
        else:
            image_path = list(self.image_dir.glob("*.png"))[idx]
            x_0 = (self.ddpm.prepare_single_image(str(image_path))).unsqueeze(0)
            x_t = self.ddpm.forward_diffusion_all_steps(image_path, specific_timesteps=self.specified_timesteps)
        return x_t
    
def create_dataloaders(dataset, batch_size=32, num_workers=4):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
        
