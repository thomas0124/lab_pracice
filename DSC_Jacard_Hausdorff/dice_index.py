import numpy as np
import glob
import pandas as pd
from PIL import Image

def load(file):
    img = Image.open(file)
    data = np.array(img).astype("float16")
    data_reshape = data.reshape(-1)
    return data_reshape

def dice(x, y):
    """
    Dice's coefficient
    Sørensen-Dice coefficient
    https://en.wikipedia.org/wiki/Dice%27s_coefficient
    """
    return 2 * np.sum(((x==y)&(x==1)&(y==1)) / float(sum(map(len, (x[x==1], y[y==1])))))

y_true = glob.glob("image1/image1.png")
y_pred = glob.glob("image2/*.png")

filename1, filename2, dice_index = [], [], []

for i in range(len(y_pred)):
    img1 = load(y_true[0])
    img2 = load(y_pred[i])
    print(img1)
    img1[img1>0]=1
    img2[img2>0]=1
    di = dice(img1,img2)
    filename1.append(y_true[0])
    filename2.append(y_pred[i])
    dice_index.append(di)
    print("{} vs {}, Dice係数:{:.6f}".format(y_true[0], y_pred[i],di))
    
df = pd.DataFrame({"File1":filename1, "File2":filename2, "Dice_Index":dice_index})
df.to_csv("result_diceindex.csv",index=False)
