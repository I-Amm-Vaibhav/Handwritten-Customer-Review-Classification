import pandas as pd
import numpy as np
import cv2


data = pd.read_csv(r"data.csv").astype('float32')

X = data.drop('0', axis=1)
y = data['0']

x = np.reshape(X.values, (X.shape[0], 28, 28))

for i in range(80, 85):
    cv2.imshow(f'{i}', x[i])
    cv2.waitKey(0)

# print(x.shape)
