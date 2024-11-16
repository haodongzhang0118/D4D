import argparse
from trainer import Trainer, ConfigParser
from model import NoiseEstimationClip
from dataset import NoiseEstimationDataset, create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = ConfigParser.parse_yaml(args.config)
    
    dataset = NoiseEstimationDataset(image_dir=config.image_dir, 
                                     clean_image=config.clean_image, 
                                     img_size=config.img_size, 
                                     specific_timesteps=config.specific_timesteps, 
                                     saved_all_data_first=config.saved_all_data_first)
    
    dataloader = create_dataloaders(dataset, 
                                    batch_size=config.batch_size, 
                                    num_workers=config.num_workers)
    
    model = NoiseEstimationClip(d_model=config.d_model,
                                in_channels=config.in_channels, 
                                image_size=config.image_size, 
                                patch_size=config.patch_size, 
                                num_heads=config.num_heads, 
                                num_layers=config.num_layers,
                                final_embedding_dim=config.final_embedding_dim)
    
    trainer = Trainer(model, dataloader, config)
    trainer.train()
    