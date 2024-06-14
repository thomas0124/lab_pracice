import torch
import torchvision.transforms as transforms
from PIL import Image

def dice_coefficient(pred, target):
    """Calculate the Dice Coefficient between two PyTorch tensors."""
    smooth = 1e-6  # To avoid division by zero
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice.item()

def preprocess_image(image_path):
    """Load an image, convert to grayscale, and threshold to binary."""
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert image to grayscale
        transforms.ToTensor(),   # Convert image to PyTorch tensor
        transforms.Lambda(lambda x: x > 0.5)  # Thresholding to convert to binary
    ])
    binary_image = transform(image).float()
    return binary_image

# Paths to the images
image_path1 = './1875081613.jpg'
image_path2 = './1576937358.jpg'

# Preprocess images
img1 = preprocess_image(image_path1)
img2 = preprocess_image(image_path2)

# Ensure the images have the same shape
if img1.shape != img2.shape:
    raise ValueError("Input images must have the same dimensions")

# Calculate Dice coefficient
dice = dice_coefficient(img1, img2)
print(f"Dice Coefficient: {dice}")
