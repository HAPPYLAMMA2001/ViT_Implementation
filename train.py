import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

from dataset import CIFAR10Dataset, get_transforms
from config import get_weights_path, get_config
from model import build_vit

from pathlib import Path

def get_dataloader(config):
    train_transform, val_transform = get_transforms(config['image_size'])
    
    # Initialize datasets with CIFAR-10
    train_dataset = CIFAR10Dataset(
        root_dir=config['dataset_dir'],
        train=True,
        transform=train_transform
    )
    val_dataset = CIFAR10Dataset(
        root_dir=config['dataset_dir'],
        train=False,
        transform=val_transform
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader

def get_model(config):
    model = build_vit(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        N=config['n_layers'],
        h=config['n_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    )
    return model

def train_model(config):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Create model directory
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # Get dataloaders
    train_dataloader, val_dataloader = get_dataloader(config)
    
    # Initialize model
    model = get_model(config).to(device)
    
    # Setup tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()
    
    # Initialize training state
    initial_epoch = 0
    global_step = 0
    
    # Load checkpoint if specified
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']

    # Training loop
    for epoch in range(initial_epoch, config['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update metrics
            train_loss += loss.item()
            batch_iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*train_correct/train_total:.2f}%"
            })
            
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Accuracy/train', 100.*train_correct/train_total, global_step)
            global_step += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_dataloader)
        val_accuracy = 100.*val_correct/val_total
        
        print(f"\nValidation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Save checkpoint
        model_filename = get_weights_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)




