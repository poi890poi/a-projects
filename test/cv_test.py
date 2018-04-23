import cv2
import numpy as np

img = np.zeros((160, 160, 3), dtype=np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('./img.jpg', img)