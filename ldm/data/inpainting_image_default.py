import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
# import albumentations
# import PIL
import numpy as np
# import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
# from functools import partial
from PIL import Image
# from tqdm import tqdm
# import lmdb
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, Subset

class InpaintingTrain(Dataset):
    def __init__(self, size, data_root, config=None):
        self.size = size
        self.config = config or OmegaConf.create()
        self.image_flist = self.get_files_from_txt(data_root)


    def generate_stroke_mask(self, im_size, parts=4, maxVertex=25, maxLength=80, maxBrushWidth=40, maxAngle=360):
        
        mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
        for i in range(parts):
            mask = mask + self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
        mask = np.minimum(mask, 1.0)

        return mask


    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):

        mask = np.zeros((h, w, 1), np.float32)
        numVertex = np.random.randint(maxVertex + 1)
        startY = np.random.randint(h)
        startX = np.random.randint(w)
        brushWidth = 0
        for i in range(numVertex):
            angle = np.random.randint(maxAngle + 1)
            angle = angle / 360.0 * 2 * np.pi
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(maxLength + 1)
            brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
            nextY = startY + length * np.cos(angle)
            nextX = startX + length * np.sin(angle)
            nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(int)
            nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(int)
            cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
            cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
            startY, startX = nextY, nextX
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        
        return mask


    def get_files_from_txt(self, path):

        file_list = []
        f = open(path)
        for line in f.readlines():
            line = line.strip("\n")
            file_list.append(line)
            sys.stdout.flush()
        f.close()

        return file_list


    def get_files(self, path):

        # read a folder, return the complete path
        ret = []
        for root, dirs, files in os.walk(path):
            for filespath in files:
                ret.append(os.path.join(root, filespath))

        return ret


    def __len__(self):
        return len(self.image_flist)


    def __getitem__(self, i):
        
        image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        
        # generate random rectangle mask with variation of width and height between 30 and 60 pixels and ratio of the input image between 20% and 70%
        mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
        min_y = random.randint(0, self.size - 60)
        min_x = random.randint(0, self.size - 60)
        max_y = random.randint(min_y + 30, self.size)
        max_x = random.randint(min_x + 30, self.size)
        mask[min_y:max_y, min_x:max_x, :] = 1

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1 - mask) * image

        # Create an image with only the masked region filled by the original pixels
        opposite_masked_image = mask * image

        batch = {"image": image, "mask": mask, "masked_image": masked_image, "opposite_masked_image": opposite_masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        # Save images in two rows using plt
        plt.figure(figsize=(12, 8))

        # First row: Original Image, Generated Mask, Masked Image
        plt.subplot(2, 3, 1)
        plt.imshow(image.numpy())
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(mask.squeeze().numpy(), cmap='gray')
        plt.title('Generated Mask')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(masked_image.numpy())
        plt.title('Masked Image')
        plt.axis('off')

        # Second row: Original Image, Generated Mask, Opposite Masked Image
        plt.subplot(2, 3, 4)
        plt.imshow(image.numpy())
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(mask.squeeze().numpy(), cmap='gray')
        plt.title('Generated Mask')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(opposite_masked_image.numpy())
        plt.title('Opposite Masked Image')
        plt.axis('off')

        plt.savefig(f"sample_images_rows_{i}.png")
        plt.close()

        return batch