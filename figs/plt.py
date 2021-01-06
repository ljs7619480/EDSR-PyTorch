import matplotlib.pyplot as plt
import cv2

lr = cv2.imread('../datasets/DIV2K/DIV2K_test_lr_unknown/11.png')
edsr = cv2.imread('../experiment/x3_Dila_ensemble/results-Demo/11_x3_SR.png')
lh, lw, _ = lr.shape
h, w, _ = edsr.shape

# fig, axs = plt.subplots(2, 3, num='Result X3', figsize=(10, 9))
# fig.suptitle('X3, {}x{} -> {}x{}'.format(lw, lh, w, h))

# axs[0, 0].set_title('Original')
# axs[0, 0].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))

# axs[0, 1].set_title('NEAREST')
# axs[0, 1].imshow(cv2.cvtColor(cv2.resize(
#     lr, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))

# axs[0, 2].set_title('LINEAR')
# axs[0, 2].imshow(cv2.cvtColor(cv2.resize(
#     lr, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))

# axs[1, 0].set_title('CUBIC')
# axs[1, 0].imshow(cv2.cvtColor(cv2.resize(
#     lr, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))

# axs[1, 1].set_title('AREA')
# axs[1, 1].imshow(cv2.cvtColor(cv2.resize(
#     lr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))

# axs[1, 2].set_title('EDSR')
# axs[1, 2].imshow(cv2.cvtColor(edsr, cv2.COLOR_BGR2RGB))


fig, axs = plt.subplots(1, 3, num='Result X3', figsize=(10, 5))
fig.suptitle('X3, {}x{} -> {}x{}'.format(lw, lh, w, h))

axs[0].set_title('Original')
axs[0].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))

axs[1].set_title('CUBIC')
axs[1].imshow(cv2.cvtColor(cv2.resize(
    lr, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))

axs[2].set_title('EDSR')
axs[2].imshow(cv2.cvtColor(edsr, cv2.COLOR_BGR2RGB))

fig.savefig('./Result_report.png')
