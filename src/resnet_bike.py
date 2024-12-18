import torch
from torchvision import models, transforms
from PIL import Image
import requests
import os

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize the image to 256 pixels (shortest side)
    transforms.CenterCrop(224),           # Center crop to 224x224 pixels
    transforms.ToTensor(),                # Convert image to tensor
    transforms.Normalize(                 # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Path to the image (ensure this path is correct)
image_path = r"C:\Users\kavit\Downloads\bike_image.jpeg"

# Check if the file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    outputs = model(input_tensor)

# Get the top-3 predictions
_, indices = outputs.topk(3)  # Top 3 predictions
indices = indices[0].tolist()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()  # Fetch labels from URL

# Print the top-3 predictions
print("Top-3 Predictions:")
for rank, idx in enumerate(indices, start=1):
    label = labels[idx]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][idx].item()
    print(f"Rank {rank}: {label} (Confidence: {confidence:.3f})")
