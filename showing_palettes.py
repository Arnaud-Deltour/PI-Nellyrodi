import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv("data_abstract2.csv")

palettes = df.values.astype(np.float32)
print(palettes[0])

#palettes[5882] = [0.19338895, 0.69179638, 0.2257361, 0.49838022, 0.77258059, 0.72849189, 0.37038655, 0.3530084,  0.62356783, 0.33628229, 0.74096639, 0.13955026]

pal = palettes[1]*255
pal1 = np.clip(pal,0,255).astype(int).reshape(4, 3).reshape(1, 4, 3)
pal_lab = cv2.cvtColor(np.array(pal1, dtype=np.uint8), cv2.COLOR_LAB2RGB)

plt.imshow(pal_lab)
plt.show()