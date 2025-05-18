# Vision Transformer (ViT) Implementation

A PyTorch implementation of the Vision Transformer (ViT) model for image classification. This implementation includes training and inference on the CIFAR-10 dataset.

## Overview

Vision Transformers (ViT) were introduced in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. This implementation provides a simplified version of the ViT architecture for educational purposes.

<p align="center">
  <img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png" width="600" alt="Vision Transformer Architecture"/>
</p>

## Features

- Complete Vision Transformer architecture implementation
- Training pipeline for CIFAR-10 dataset
- Patch embedding and positional encoding
- Multi-head self-attention mechanism
- Visualization support using TensorBoard
- Inference code for classifying new images

## Model Architecture

The Vision Transformer implementation includes:

- **Patch Embedding**: Splits images into fixed-size patches and projects them into an embedding space
- **Position Encoding**: Adds positional information to the patch embeddings
- **Transformer Encoder**: Multiple layers of self-attention and feed-forward networks
- **Classification Head**: MLP layer for final classification

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- PIL
- numpy
- tqdm
- tensorboard

## Installation

```bash
# Clone the repository
git clone https://github.com/HAPPYLAMMA2001/ViT_Implementation.git
cd ViT_Implementation

# Install dependencies
pip install torch torchvision tqdm numpy pillow tensorboard
```

## Dataset

The implementation uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Usage

### Configuration

Edit `config.py` to adjust hyperparameters, model architecture, and training settings.

### Training

```bash
python train.py
```

### Inference

```bash
python predict.py
```

To use your own image for inference, update the `file_path` in `predict.py`:

```python
file_path = "path/to/your/image.jpg"  # Replace with the path to your image
```

## Model Configuration

The default configuration:

- Image size: 224x224
- Patch size: 16x16
- Embedding dimension: 384
- Number of heads: 6
- Number of layers: 7
- MLP hidden dimension: 1536
- Dropout rate: 0.1

## Project Structure

- `model.py`: Vision Transformer model implementation
- `train.py`: Training script with dataloaders and training loop
- `dataset.py`: Dataset classes for CIFAR-10 and custom image datasets
- `config.py`: Configuration parameters for the model and training
- `predict.py`: Inference script for classifying new images

## Results

The model achieves competitive performance on the CIFAR-10 test set with appropriate hyperparameter tuning.

## Acknowledgements

- The original [Vision Transformer paper](https://arxiv.org/abs/2010.11929)
- PyTorch documentation and tutorials
- CIFAR-10 dataset creators
