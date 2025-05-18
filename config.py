from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-3,
        # ViT specific parameters
        "image_size": 224,  # ViT standard input size
        "patch_size": 16,   # Size of patches to be extracted from image
        "in_channels": 3,   # RGB images
        "num_classes": 10,  # CIFAR-10 has 10 classes
        "d_model": 384,     # Hidden dimension
        "n_heads": 6,      # Number of attention heads
        "n_layers": 7,     # Number of transformer layers
        "dropout": 0.1,    # Dropout rate
        "d_ff": 1536,      # Feed-forward dimension (4 * d_model)
        # Dataset parameters
        "dataset_dir": "dataset/cifar-10-batches-py",
        "model_folder": "weights",
        "model_filename": "vit_cifar10_",
        "preload": None,
        "experiment_name": "runs/vit_cifar10",
        "class_to_idx": {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }
        
    }

def get_weights_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder/model_filename)