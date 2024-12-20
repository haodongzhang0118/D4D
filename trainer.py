import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from einops import rearrange
from types import SimpleNamespace
from typing import Dict, Any
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from utils.DDPMForward import DDPMForward

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
            valid_data_loader,
            config,
    ):
        self._validate_config(config)
        
        # Initialize basic attributes
        self.model = model
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ddpm = DDPMForward(transform=None, num_timesteps=config.num_timesteps, img_size=config.image_size)
        self.text_image = config.text_image
        self.model.to(self.device)

        # Setup optimizer and schedulers
        self._setup_optimizer_and_scheduler()
        
        # Setup logging
        self.setup_logging()

        # Initialize tracking variables
        self.best_valid_acurracy = 0
        self.current_epoch = 0

        # Load checkpoint if specified
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        # Optionally plot initial learning rate schedule
        if self.config.use_wandb:
            self.plot_lr_schedule()

    def _validate_config(self, config):
        assert config.warmup_epochs >= 0, "warmup_epochs must be >= 0"
        assert config.warmup_epochs < config.epochs, "warmup_epochs must be < epochs"
        assert config.min_lr <= config.lr, "min_lr must be <= lr"
        assert config.batch_size > 0, "batch_size must be > 0"

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
        )

        scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.min_lr
        )

        self.scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1.0,
            total_epoch=self.config.warmup_epochs,
            after_scheduler=scheduler_cosine
        )

        self.optimizer.zero_grad()
        self.optimizer.step()
        if self.config.warmup_epochs == 0:
            self.scheduler.step()

    def setup_logging(self):
        """Setup logging directories and wandb"""
        self.log_dir = Path(self.config.log_dir) / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize wandb if enabled
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
            'best_valid_acurracy': self.best_valid_acurracy,
            'config': self.config
        }

        checkpoint_path = self.log_dir / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.log_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            logging.info(f'Saved new best model with accuracy {self.best_valid_acurracy:.4f}')

    def load_checkpoint(self, path):
        """Load model and training state from checkpoint"""
        logging.info(f'Loading checkpoint from {path}')
        checkpoint = torch.load(path)
        
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.config.resetLR:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.lr
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.current_epoch,
                eta_min=self.config.min_lr
            )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_valid_acurracy = checkpoint['best_valid_acurracy']
        
        logging.info(f'Loaded checkpoint from epoch {self.current_epoch} '
                    f'with best loss {self.best_valid_acurracy:.4f}')

    @torch.no_grad()
    def valid_accuracy_calculate(self):
        self.model.eval()
        predictions = []

        for images, label, specific in self.valid_data_loader:
            images = images.to(self.device)
            label = label.to(self.device)
            _, C, H, W = images.shape
            t = torch.zeros((1, 1, C, H, W))
            t = (self.ddpm.forward_diffusion(t, specific[0].item()).squeeze(0)).to(self.device)
            logits = self.model(images, t)
            logits = (100 * logits).softmax(dim=-1)
            predictions.append((torch.argmax(logits, dim=1) == label))
        
        predictions = torch.cat(predictions)
        return torch.mean(predictions.float()).item()
    
    @torch.no_grad()
    def validation_step(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, label, specific in self.valid_data_loader:
                images = images.to(self.device)
                timesteps = torch.arange(specific[0].item()).to(self.device)
                
                logits = self.model(images, timesteps)
                pred = torch.argmax(logits, dim=1)
                
                correct += (pred == label.to(self.device)).sum().item()
                total += label.size(0)
                
        return correct / total

        

    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        current_lr = self.scheduler.get_last_lr()[0]
        
        logging.info(f'Current learning rate: {current_lr:.8f}')
        pbar = tqdm(self.data_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, inputx in enumerate(pbar):
            inputx = rearrange(inputx, 'b t c h w -> t b c h w')
            
            indices = torch.stack(
                [torch.randperm(inputx.shape[1]) for _ in range(inputx.shape[0])]
            )
            x = torch.stack([inputx[i, indices[i]] for i in range(inputx.shape[0])])
            
            T, B, C, H, W = x.shape
            if self.text_image:
                t = torch.arange(T)
            else:
                t = torch.zeros((1, C, H, W))
                t = self.ddpm.forward_diffusion(t, T).squeeze(0)
            
            batch_loss = 0
            for i in range(B):
                images = x[:, i, :, :, :]
                images = images.to(self.device)
                t = t.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images, t)
                labels = torch.arange(len(images)).to(self.device)
                loss_i = nn.functional.cross_entropy(logits, labels)
                loss_t = nn.functional.cross_entropy(logits.T, labels)
                loss = (loss_i + loss_t) / 2.0
                
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                self.optimizer.step()
                batch_loss += loss.item()

            batch_loss = batch_loss / B
            total_loss += batch_loss
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.6f}'
            })

            if self.config.use_wandb:
                wandb.log({
                    'train_batch_loss': batch_loss,
                    'learning_rate': current_lr
                })

        return total_loss / len(self.data_loader)

    def train(self):
        logging.info(f'Starting training with config:\n{vars(self.config)}')
        logging.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters())}')
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                logging.info(f'Starting epoch {epoch}')
                
                train_loss = self.train_one_epoch()
                self.scheduler.step()

                logging.info(
                    f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'lr={self.scheduler.get_last_lr()[0]:.8f}'
                )

                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
                if self.text_image:
                    valid_accuracy = self.validation_step()
                else:
                    valid_accuracy = self.valid_accuracy_calculate()
                print(f'Validation accuracy: {valid_accuracy:.4f}')

                is_best = valid_accuracy > self.best_valid_acurracy
                if is_best:
                    self.best_valid_acurracy = valid_accuracy
                self.save_checkpoint(is_best=is_best)

        except KeyboardInterrupt:
            logging.info('Training interrupted by user')
            self.save_checkpoint()
        
        except Exception as e:
            logging.error(f'Training failed with error: {str(e)}')
            self.save_checkpoint()
            raise e

        finally:
            if self.config.use_wandb:
                wandb.finish()
            
            logging.info(f'Training finished.')