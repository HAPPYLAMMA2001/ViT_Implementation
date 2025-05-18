from PIL import Image
import torch
from torchvision import transforms
from config import get_config, get_weights_path
from train import get_model

def predict_image(file_path, config):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = get_model(config).to(device)
    state = torch.load("final/vit_cifar10_17.pt", map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    # Define preprocessing (same as in dataset)
    preprocess = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Load and preprocess the image
    image = Image.open(file_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    # Map predicted label to class name
    class_names = sorted([cls for cls in config['class_to_idx']])
    predicted_label = class_names[predicted.item()]
    
    return predicted_label

# Example usage
if __name__ == "__main__":
    config = get_config()
    file_path = r"C:\Users\shami\Downloads\frog.jpeg" # Replace with the path to your image
    label = predict_image(file_path, config)
    print(f"Predicted Label: {label}")