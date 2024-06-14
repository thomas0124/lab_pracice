import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.spatial.distance import directed_hausdorff
# from skimage.measure import label


def dice_coefficient(img1, img2):
    """Calculate the Dice Coefficient between two binary images."""
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.sum(img1 & img2)
    return 2. * intersection / (np.sum(img1) + np.sum(img2))

def jaccard_index(img1, img2):
    """Calculate the Jaccard Index between two binary images."""
    img1 = img1.astype(bool)
    img2 = img2.astype(bool)
    intersection = np.sum(img1 & img2)
    union = np.sum(img1 | img2)
    return intersection / union

def hausdorff_distance(img1, img2):
    """Calculate the Hausdorff Distance between two binary images."""
    img1_points = np.argwhere(img1)
    img2_points = np.argwhere(img2)
    forward_hausdorff = directed_hausdorff(img1_points, img2_points)[0]
    backward_hausdorff = directed_hausdorff(img2_points, img1_points)[0]
    return max(forward_hausdorff, backward_hausdorff)

# Load and preprocess the images
def preprocess_image(image_path):
    """Load an image, convert to grayscale, and threshold to binary."""
    image = imread(image_path)
    if image.ndim == 3:  # Convert RGB to grayscale if necessary
        image = rgb2gray(image)
    binary_image = image > 0.5  # Thresholding to convert to binary
    return binary_image

# Paths to the images
image_path1 = './1875081613.jpg'
image_path2 = './1576937358.jpg'

# Preprocess images
img1 = preprocess_image(image_path1)
img2 = preprocess_image(image_path2)

# Calculate similarity metrics
dice = dice_coefficient(img1, img2)
jaccard = jaccard_index(img1, img2)
hausdorff = hausdorff_distance(img1, img2)

print(f"Dice Coefficient: {dice}")
print(f"Jaccard Index: {jaccard}")
print(f"Hausdorff Distance: {hausdorff}")
