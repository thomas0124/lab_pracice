import numpy as np
import glob
import pandas as pd
from PIL import Image

def load(file):
    img = Image.open(file)
    data = np.array(img).astype("float16")
    data_reshape = data.reshape(-1)
    return data_reshape

def jaccard(x, y):
    intersection = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))
    return intersection / union

y_true = glob.glob("image1/image1.png")
y_pred = glob.glob("image2/*.png")

filename1, filename2, jaccard_index = [], [], []

for i in range(len(y_pred)):
    img1 = load(y_true[0])
    img2 = load(y_pred[i])
    img1[img1>0]=1
    img2[img2>0]=1
    ja = jaccard(img1,img2)
    filename1.append(y_true[0])
    filename2.append(y_pred[i])
    jaccard_index.append(ja)
    print("{} vs {}, Jaccard係数:{:.6f}".format(y_true[0], y_pred[i],ja))
    
df = pd.DataFrame({"File1":filename1, "File2":filename2, "Jaccard_Index":jaccard_index})
df.to_csv("result_jaccardindex.csv",index=False)
