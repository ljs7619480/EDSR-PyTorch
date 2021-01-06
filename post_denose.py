import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

img_dir = 'experiment/x3_Dila_UNK_ens/results-Demo'
out_dir = os.path.join(img_dir, 'denose')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img_names = os.listdir(img_dir)
for img_name in img_names:
    if not img_name.endswith('.png'):
        continue
    img = cv2.imread(os.path.join(img_dir, img_name))
    dst = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    cv2.imwrite(os.path.join(out_dir, img_name), dst)
    # plt.subplot(121), plt.imshow(cv2.cvtColor(
    #     img, cv2.COLOR_BGR2RGB))
    # plt.subplot(122), plt.imshow(cv2.cvtColor(
    #     dst, cv2.COLOR_BGR2RGB))
    # plt.show()
