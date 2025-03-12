import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model_from_checkpoint(checkpoint_path, num_classes, device='cpu'):
    """
    Loads a ResNet-50 model from a checkpoint file with the final FC layer adjusted.
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_single_image(image_path, checkpoint_path, num_classes, device='cpu'):
    """
    Loads the best model from a checkpoint and predicts the class for a single input image.
    
    Args:
        image_path (str): Path to the input image file.
        checkpoint_path (str): Path to the checkpoint file.
        num_classes (int): Number of classes for the final FC layer.
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        int: Predicted class index.
    """
    model = load_model_from_checkpoint(checkpoint_path, num_classes, device=device)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(input_batch.to(device))
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

if __name__ == '__main__':
    image_path = '../data/split_ttv_dataset_type_of_plants/images/test/watermelon/aug_0_330.jpg'
    checkpoint_path = './split_plants_checkpoints/best_model_epoch44.pth'
    num_classes = 30 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    classes = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']
    predicted_class = predict_single_image(image_path, checkpoint_path, num_classes, device)
    print(f"Predicted class index: {classes[predicted_class]}")
