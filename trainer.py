import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from einops import rearrange
from types import SimpleNamespace
from typing import Dict, Any

class ConfigParser:
    @staticmethod
    def parse_yaml(yaml_path: str) -> SimpleNamespace:
        """Parse YAML config file to namespace object"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return ConfigParser.dict_to_namespace(config_dict)
    
    @staticmethod
    def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
        """Recursively convert dictionary to namespace object"""
        namespace = SimpleNamespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, ConfigParser.dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

class Trainer:
    def __init__(
            self,
            model,
            data_loader,
            config,
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, T_max=self.config.epochs, eta_min=config.min_lr)
        self.setup_logging()

        self.best_loss = float('inf')
        self.current_epoch = 0

        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
    
    def setup_logging(self):
        self.log_dir = Path(self.config.log_dir) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                config=vars(self.config)
            )

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, self.log_dir / 'checkpoint.pth')
        if is_best:
            torch.save(checkpoint, self.log_dir / 'best.pth')

    def load_checkpoint(self, path):
        logging.info(f'Loading checkpoint from {path}')
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        logging.info(f'Loaded checkpoint from {path} (epoch {self.current_epoch})')

    def train_one_epoch(self, inputx):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.data_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, inputx in enumerate(pbar):
            # x size: B, T, C, H, W
            inputx = rearrange(inputx, 'b t c h w -> t b c h w')
            indices = torch.stack([torch.randperm(inputx.shape[1]) for _ in range(inputx.shape[0])])  # Shape: [256, 16]
            x = torch.stack([inputx[i, indices[i]] for i in range(inputx.shape[0])])
            x = x.to(self.device)
            T, B, _, _, _ = x.shape
            t = torch.arange(T).to(self.device)
            batch_loss = 0
            for i in range(B):
                images = x[:, i, :, :, :]
                logits = self.model(images, t)

                self.optimizer.zero_grad()

                labels = torch.arange(len(images)).to(self.device)
                loss_i = nn.functional.cross_entropy(logits, labels)
                loss_t = nn.functional.cross_entropy(logits.t(), labels)
                loss = (loss_i + loss_t) / 2

                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
            total_loss += batch_loss / B
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'},
                             {'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'})
            
            if self.config.use_wandb:
                wandb.log({'train_batch_loss': batch_loss / B},
                          {'learning_rate':self.scheduler.get_last_lr()[0]})
        return total_loss / len(self.data_loader)
    
    def train(self):
        logging.info(f'Starting training with config:\n{vars(self.config)}')

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            logging.info(f'Starting epoch {epoch}')
            train_loss = self.train_one_epoch(self.data_loader)
            self.scheduler.step()
            logging.info(
                f'Epoch {epoch}: train loss={train_loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.6f}')

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint(is_best=True)
            self.save_checkpoint()

            self.save_checkpoint()

        logging.info('Training finished')
