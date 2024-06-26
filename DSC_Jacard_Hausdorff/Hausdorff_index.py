import numpy as np
import glob
import pandas as pd
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

def load(file):
    img = Image.open(file)
    data = np.array(img).astype("float16")
    return data

def uniform_sample_points(points, sample_size):
    """
    Uniformly sample points from a set of coordinates.
    """
    if len(points) <= sample_size:
        return points
    # Determine the step size for uniform sampling
    step = len(points) // sample_size
    return points[::step]

def hausdorff_distance(mask1, mask2, sample_size=None):
    """
    Calculate the Hausdorff distance between two binary masks.
    Optionally, use a sample of the points for faster computation.
    """
    # Get coordinates of non-zero pixels
    points1 = np.argwhere(mask1)
    points2 = np.argwhere(mask2)
    
    if sample_size:
        points1 = uniform_sample_points(points1, sample_size)
        points2 = uniform_sample_points(points2, sample_size)
    
    # Compute directed Hausdorff distances between the point sets
    hausdorff_distance_1_to_2 = directed_hausdorff(points1, points2)[0]
    hausdorff_distance_2_to_1 = directed_hausdorff(points2, points1)[0]
    
    # Take the maximum distance (symmetric Hausdorff distance)
    hausdorff_distance = max(hausdorff_distance_1_to_2, hausdorff_distance_2_to_1)
    
    return hausdorff_distance

y_true = glob.glob("image1/image1.png")
y_pred = glob.glob("image2/*.png")

filename1, filename2, hausdorff_distances = [], [], []

sample_size = 200000  # サンプルサイズを設定（例：10000）

for i in range(len(y_pred)):
    img1 = load(y_true[0])
    img2 = load(y_pred[i])
    hd = hausdorff_distance(img1, img2, sample_size)
    filename1.append(y_true[0])
    filename2.append(y_pred[i])
    hausdorff_distances.append(hd)
    print("{} vs {}, ハウスドルフ距離:{:.6f}".format(y_true[0], y_pred[i], hd))

df = pd.DataFrame({"File1":filename1, "File2":filename2, "Hausdorff_Distance":hausdorff_distances})
df.to_csv("result_hausdorff.csv",index=False)
