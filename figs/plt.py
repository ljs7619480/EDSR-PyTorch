import matplotlib.pyplot as plt
import cv2

lr = cv2.imread('../datasets/DIV2K/DIV2K_test_lr_unknown/11.png')
edsr = cv2.imread('../experiment/x3_Dila_ensemble/results-Demo/11_x3_SR.png')
lh, lw, _ = lr.shape
h, w, _ = edsr.shape
bicubic = cv2.resize(lr, (w, h))

fig, axs = plt.subplots(1, 2, num='Result X3')
fig.suptitle('X3, {}x{} -> {}x{}'.format(lw, lh, w, h))
axs[0].set_title('BICUBIC')
axs[0].imshow(cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB))

axs[1].set_title('EDSR')
axs[1].imshow(cv2.cvtColor(edsr, cv2.COLOR_BGR2RGB))

fig.savefig('./Result.png')
