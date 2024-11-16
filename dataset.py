import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils.DDPMForward import DDPMForward

class NoiseEstimationDataset(Dataset):
    def __init__(self, image_dir, 
                 transform=None, 
                 clean_image=None, 
                 img_size=256, 
                 num_timesteps=1000, 
                 specific_timesteps=256, 
                 saved_all_data_first=False,
                 num_cores=6):
        
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.specified_timesteps = specific_timesteps
        self.transform = transform
        self.ddpm = DDPMForward(transform=self.transform, num_timesteps=num_timesteps, img_size=img_size)
        self.saveFirst = saved_all_data_first
        if self.saveFirst:
            # Load All image to torch tensor first to save Dataloader time. It will generate a pt file called cleanImage.pt which contains this tensor
            self.x_0 = self.ddpm.Load_all_images(image_dir=image_dir, clean_image=clean_image, num_cores=num_cores) # About 32 GB of data

    def __len__(self):
        if self.saveFirst:
            return self.x_0.shape[0]
        else:
            return len(list(self.image_dir.glob("*.png")))

    def __getitem__(self, idx):
        if self.saveFirst:
            x_0 = self.x_0[idx]
            x_t = self.ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=self.specified_timesteps)
        else:
            image_path = list(self.image_dir.glob("*.png"))[idx]
            x_0 = (self.ddpm.prepare_single_image(str(image_path))).unsqueeze(0)
            x_t = self.ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=self.specified_timesteps)
        x_t = x_t.squeeze(0)
        return x_t

class NoiseEstimationValidationDataset(Dataset):
    def __init__(self, image_dir, 
                 transform=None, 
                 clean_image=None, 
                 img_size=256, 
                 num_timesteps=1000, 
                 specific_timesteps=256, 
                 saved_all_data_first=False,
                 num_cores=6):
        
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.specified_timesteps = specific_timesteps
        self.transform = transform
        self.ddpm = DDPMForward(transform=self.transform, num_timesteps=num_timesteps, img_size=img_size)
        self.saveFirst = saved_all_data_first
        if self.saveFirst:
            # Load All image to torch tensor first to save Dataloader time. It will generate a pt file called cleanImage.pt which contains this tensor
            self.x_0 = self.ddpm.Load_all_images(image_dir=image_dir, clean_image=clean_image, num_cores=num_cores) # About 32 GB of data

    def __len__(self):
        if self.saveFirst:
            return self.x_0.shape[0]
        else:
            return len(list(self.image_dir.glob("*.png")))

    def __getitem__(self, idx):
        if self.saveFirst:
            x_0 = self.x_0[idx]
            x_t = self.ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=self.specified_timesteps)
        else:
            image_path = list(self.image_dir.glob("*.png"))[idx]
            x_0 = (self.ddpm.prepare_single_image(str(image_path))).unsqueeze(0)
            x_t = self.ddpm.forward_diffusion_all_steps(x_0, specific_timesteps=self.specified_timesteps)
        label = torch.randint(0, self.specified_timesteps, (1,)).item()
        x_t = x_t.squeeze(0)
        x_t = x_t[label]
        return x_t, label

def create_dataloaders(dataset, batch_size=32, num_workers=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return dataloader
