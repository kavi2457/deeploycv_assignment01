import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Load the pre-trained ResNet-50 model with the updated weights argument
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize the image to have a minimum dimension of 256 pixels
    transforms.CenterCrop(224),  # Crop the image to 224x224 around the center
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load and preprocess the bike image
image_path = r"../images/bike_image.jpeg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Convert to RGB to avoid issues with grayscale or other formats
input_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension

# Perform inference
with torch.no_grad():  # Disable gradient computation
    outputs = model(input_tensor)

# Get the top-3 predictions
_, indices = outputs.topk(3)  # Get the indices of the top-3 predictions
indices = indices[0].tolist()  # Convert to a list for easier use

# Load the labels for ImageNet classes
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()  # Fetch the labels from the URL

# Print the top-3 predictions with their confidence scores
for idx in indices:
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][idx].item()  # Convert raw scores to probabilities
    print(f"Prediction: {labels[idx]}, Confidence: {confidence:.3f}")
