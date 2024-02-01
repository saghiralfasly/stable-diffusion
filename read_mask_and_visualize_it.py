import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

# # # Load the mask from .npy file
mask = np.load("/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks/tumor_095/tumor_095_21122_86278.npy")
# mask = np.load("/mayo_atlas/atlas/publicDatasets/PANDA/57fca8dcbc2f311382970f20842a923c_18616_27924.npy")
print(mask)
mask = mask.astype(np.uint8)

print(mask.shape)
# print(mask)
# print(mask.tolist())

# # read the corresponding image using Image PIL
# # img = Image.open("/mayo_atlas/atlas/publicDatasets/PANDA/57fca8dcbc2f311382970f20842a923c_18616_27924.png")
# img = Image.open("/mayo_atlas/atlas/publicDatasets/PANDA/57fca8dcbc2f311382970f20842a923c.png")
# # save the image as a PNG image
# img.save('mask_image_read1.png')



# onehot = np.eye(2)[mask]
# # print(onehot.tolist())
# # print(onehot.shape)
# onehot = onehot.transpose(2, 0, 1)
# print(onehot.tolist())
# # print(onehot.shape)

# save it again to mask_read2.png
plt.imsave('mask_read_new.png', mask, cmap='gray')
        
# print(type(mask))
# print(mask)
# print(mask.shape)

# print(type(mask2))
# print(mask2)
# print(mask2.shape)

# Save the mask as a PNG image
# plt.imsave('mask_read.png', mask)
# plt.imsave('mask_read2.png', mask, cmap='gray')
# Save the mask as a PNG image
# plt.imsave('mask_read2.png', mask2)
# plt.imsave('mask_read2.png', mask, cmap='gray')

# # read the mask
# # mask = plt.imread('mask_read2.png')
# # read the image in mode == "L" using Image PIL
# # mask = Image.open('mask_read2.png').convert('L')


# mask = cv2.imread('mask_read2.png', cv2.IMREAD_GRAYSCALE)

# # mask = plt.imread('mask_read2.png', mode="L")
# print(mask.shape)
# print(mask)
# # save it again to mask_read2.png
# # plt.imsave('mask_read2.png', mask, cmap='gray')