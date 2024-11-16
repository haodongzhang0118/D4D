import argparse
from trainer import Trainer, ConfigParser
from model import NoiseEstimationClip
from dataset import NoiseEstimationDataset, create_dataloaders
from types import SimpleNamespace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()

def flatten_namespace(nested_namespace):
    flat_namespace = SimpleNamespace()
    def add_attributes(ns):
        for key, value in vars(ns).items():
            if isinstance(value, SimpleNamespace):
                add_attributes(value)
            else:
                setattr(flat_namespace, key, value)
    
    add_attributes(nested_namespace)
    return flat_namespace

if __name__ == '__main__':
    args = parse_args()
    config = ConfigParser.parse_yaml(args.config)
    config = flatten_namespace(config)
    
    dataset = NoiseEstimationDataset(image_dir=config.image_dir, 
                                     clean_image=config.clean_image, 
                                     img_size=config.image_size, 
                                     specific_timesteps=config.specific_timesteps, 
                                     saved_all_data_first=config.saved_all_data_first,
                                     num_cores=config.num_cores)
    
    valid_dataset = NoiseEstimationDataset(image_dir=config.valid_dir,
                                           clean_image=config.valid_image,
                                           img_size=config.image_size,
                                           specific_timesteps=config.specific_timesteps,
                                           saved_all_data_first=config.saved_all_data_first,
                                           num_cores=config.num_cores)
    
    dataloader = create_dataloaders(dataset, 
                                    batch_size=config.batch_size, 
                                    num_workers=config.num_workers)
    
    valid_dataloader = create_dataloaders(valid_dataset,
                                          batch_size=config.batch_size,
                                          num_workers=config.num_workers)
    
    model = NoiseEstimationClip(d_model=config.d_model,
                                in_channels=config.in_channels, 
                                image_size=config.image_size, 
                                patch_size=config.patch_size, 
                                num_heads=config.num_heads, 
                                num_layers=config.num_layers,
                                final_embedding=config.final_embedding_dim)
    
    trainer = Trainer(model, dataloader, valid_dataloader, config)
    trainer.train()
    