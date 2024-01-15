import os, sys, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
import lmdb
import random
import torch
from torch.utils.data import Dataset, Subset

import matplotlib.pyplot as plt

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

    def random_crop(self, region, percentage):
        h, w, _ = region.shape
        min_size = min(h, w)
        new_h, new_w = int(min_size * percentage), int(min_size * percentage)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        cropped = region[top: top + new_h, left: left + new_w, :]
        return cropped


    def __getitem__(self, i):
        
        image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        
        # generate random rectangle mask with variation of width and height
        mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
        min_y = random.randint(0, self.size - 400)
        min_x = random.randint(0, self.size - 400)
        max_y = random.randint(min_y + 200, self.size)
        max_x = random.randint(min_x + 200, self.size)
        mask[min_y:max_y, min_x:max_x, :] = 1
        
        # get half of the width and height of the mask
        half_width = (max_x - min_x) // 4
        half_height = (max_y - min_y) // 4
        # make the center of the mask with this half width and height as white
        mask[min_y+half_height:max_y-half_height, min_x+half_width:max_x-half_width, :] = 0
        
        # # create inner mask inside the range of min_y, min_x, max_y, max_x
        # # inner_mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
        # min_y2 = random.randint(min_y, min_y+100)
        # min_x2 = random.randint(min_x, max_x+100)
        # max_y2 = random.randint(min_y2, max_y)
        # max_x2 = random.randint(min_x2, max_x)
        # mask[min_y2:max_y2, min_x2:max_x2, :] = 0
        

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1 - mask) * image

        # Create an image with only the masked region filled by the original pixels
        # opposite_masked_image = mask * image
        
        # print(type(image))
        # print(type(mask))
        # print(type(opposite_masked_image))

        # batch = {"image": image, "mask": mask, "masked_image": masked_image, "opposite_masked_image": opposite_masked_image}
        # batch = {"image": image, "mask": mask, "masked_image": opposite_masked_image}
        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        # # Save images in two rows using plt
        # plt.figure(figsize=(12, 8))

        # # First row: Original Image, Generated Mask, Masked Image
        # plt.subplot(2, 3, 1)
        # plt.imshow(image.numpy())
        # plt.title('Original Image')
        # plt.axis('off')

        # plt.subplot(2, 3, 2)
        # plt.imshow(mask.squeeze().numpy(), cmap='gray')
        # plt.title('Generated Mask')
        # plt.axis('off')

        # plt.subplot(2, 3, 3)
        # plt.imshow(masked_image.numpy())
        # plt.title('Masked Image')
        # plt.axis('off')

        # # Second row: Original Image, Generated Mask, Opposite Masked Image
        # plt.subplot(2, 3, 4)
        # plt.imshow(image.numpy())
        # plt.title('Original Image')
        # plt.axis('off')

        # plt.subplot(2, 3, 5)
        # plt.imshow(mask.squeeze().numpy(), cmap='gray')
        # plt.title('Generated Mask')
        # plt.axis('off')

        # plt.subplot(2, 3, 6)
        # plt.imshow(opposite_masked_image.numpy())
        # plt.title('Opposite Masked Image')
        # plt.axis('off')

        # plt.savefig(f"sample_images_rows_NEw{i}.png")
        # plt.close()

        return batch
    
    # masked region roi
    # def __getitem__(self, i):
        
    #     image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
    #     image = cv2.resize(image, (self.size, self.size))
    #     image = image.astype(np.float32) / 255.0
    #     image = torch.from_numpy(image)

        
    #     # generate random rectangle mask with variation of width and height
    #     mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
    #     min_y = random.randint(0, self.size - 60)
    #     min_x = random.randint(0, self.size - 60)
    #     max_y = random.randint(min_y + 30, self.size)
    #     max_x = random.randint(min_x + 30, self.size)
    #     mask[min_y:max_y, min_x:max_x, :] = 1

    #     mask[mask < 0.5] = 0
    #     mask[mask >= 0.5] = 1
    #     mask = torch.from_numpy(mask)

    #     # masked_image = (1 - mask) * image

    #     # Create an image with only the masked region filled by the original pixels
    #     opposite_masked_image = mask * image
        
    #     # print(type(image))
    #     # print(type(mask))
    #     # print(type(opposite_masked_image))

    #     # batch = {"image": image, "mask": mask, "masked_image": masked_image, "opposite_masked_image": opposite_masked_image}
    #     batch = {"image": image, "mask": mask, "masked_image": opposite_masked_image}
    #     for k in batch:
    #         batch[k] = batch[k] * 2.0 - 1.0

    #     # # Save images in two rows using plt
    #     # plt.figure(figsize=(12, 8))

    #     # # First row: Original Image, Generated Mask, Masked Image
    #     # plt.subplot(2, 3, 1)
    #     # plt.imshow(image.numpy())
    #     # plt.title('Original Image')
    #     # plt.axis('off')

    #     # plt.subplot(2, 3, 2)
    #     # plt.imshow(mask.squeeze().numpy(), cmap='gray')
    #     # plt.title('Generated Mask')
    #     # plt.axis('off')

    #     # plt.subplot(2, 3, 3)
    #     # plt.imshow(masked_image.numpy())
    #     # plt.title('Masked Image')
    #     # plt.axis('off')

    #     # # Second row: Original Image, Generated Mask, Opposite Masked Image
    #     # plt.subplot(2, 3, 4)
    #     # plt.imshow(image.numpy())
    #     # plt.title('Original Image')
    #     # plt.axis('off')

    #     # plt.subplot(2, 3, 5)
    #     # plt.imshow(mask.squeeze().numpy(), cmap='gray')
    #     # plt.title('Generated Mask')
    #     # plt.axis('off')

    #     # plt.subplot(2, 3, 6)
    #     # plt.imshow(opposite_masked_image.numpy())
    #     # plt.title('Opposite Masked Image')
    #     # plt.axis('off')

    #     # plt.savefig(f"sample_images_rows_{i}.png")
    #     # plt.close()

    #     return batch

    
    
    
    # old
    # def __getitem__(self, i):
        
    #     image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
    #     image = cv2.resize(image, (self.size, self.size))
    #     image = image.astype(np.float32) / 255.0
    #     image = torch.from_numpy(image)

    #     # mask = self.generate_stroke_mask([self.size, self.size])
        
    #     # generate random rectangle mask with variation of width and height between 30 and 60 pixels and ratio of the input image between 20% and 70%
    #     mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
    #     min_y = random.randint(0, self.size - 60)
    #     min_x = random.randint(0, self.size - 60)
    #     max_y = random.randint(min_y + 30, self.size)
    #     max_x = random.randint(min_x + 30, self.size)
    #     mask[min_y:max_y, min_x:max_x, :] = 1

    #     # print(mask.shape)
    #     # maskIm = mask.copy()
        
        
    #     mask[mask < 0.5] = 0
    #     mask[mask >= 0.5] = 1
    #     mask = torch.from_numpy(mask)

    #     # print(f'mask.shape: {mask.shape}')
    #     # print(f'image.shape: {image.shape}')
        
    #     # # save the mask for visualization
    #     # maskIm = np.squeeze(maskIm, axis=2)
    #     # maskIm = maskIm * 255
    #     # maskIm = maskIm.astype(np.uint8)
    #     # maskIm = Image.fromarray(maskIm)
    #     # maskIm.save("/mayo_atlas/home/m288756/stable-diffusion/mask.png")

    #     # masked_image = (1 - mask) * image

        
    #     # Find bounding box of the masked region
    #     non_zero_indices = torch.nonzero(mask, as_tuple=False)
    #     if non_zero_indices.numel() > 0:
    #         min_yx = torch.min(non_zero_indices, dim=0)[0]
    #         max_yx = torch.max(non_zero_indices, dim=0)[0]
    #         min_y, min_x = min_yx[0], min_yx[1]
    #         max_y, max_x = max_yx[0], max_yx[1]

    #         # Ensure minimum width and height of 30 pixels
    #         if max_y - min_y < 30:
    #             center_y = (max_y + min_y) // 2
    #             min_y = max(center_y - 15, 0)
    #             max_y = min(center_y + 15, self.size - 1)

    #         if max_x - min_x < 30:
    #             center_x = (max_x + min_x) // 2
    #             min_x = max(center_x - 15, 0)
    #             max_x = min(center_x + 15, self.size - 1)

    #         # Crop the region from the original image
    #         cropped_region = image[min_y:max_y+1, min_x:max_x+1, :]
    #     else:
    #         # If there are no nonzero indices, crop a 30x30 region from the center of the mask location
    #         center_y, center_x = self.size // 2, self.size // 2
    #         min_y = max(center_y - 15, 0)
    #         max_y = min(center_y + 15, self.size - 1)
    #         min_x = max(center_x - 15, 0)
    #         max_x = min(center_x + 15, self.size - 1)
    #         # print("No nonzero indices crop -------------------")

    #         cropped_region = image[min_y:max_y+1, min_x:max_x+1, :]
        
    #     # Randomly crop between 50% and 100% of the cropped_region
    #     # crop_percentage = random.uniform(0.8, 1.0)
    #     # cropped_region = self.random_crop(cropped_region, crop_percentage)

    #     # now resize the cropped_region to 512x512
    #     # cropped_region2 = cv2.resize(np.array(cropped_region), (512, 512))
        
        
        
    #     # create a black image of 512x512 and paste the cropped_region2 in the masked region coordinates
    #     black_image = np.zeros((512, 512, 3), dtype=np.uint8)
    #     # Convert float32 to uint8 (integer type)
    #     black_image = (black_image * 255).astype(np.uint8)
    #     print(f"black_image.shape: {black_image.shape}")
    #     # convert the black_image to cv2 image
    #     black_image = Image.fromarray(black_image)
    #     # Calculate the box coordinates (left, top, right, bottom)
    #     box = (min_x, min_y, min_x + cropped_region.shape[1], min_y + cropped_region.shape[2])
    #     # convert the cropped_region to pil image from torch tensor
    #     cropped_region = cropped_region.permute(2, 0, 1)
    #     cropped_region = TF.to_pil_image(cropped_region)
    #     # cropped_region_new = Image.fromarray(np.array(cropped_region))
    #     print(f"cropped_region.size: {cropped_region.size}")
    #     print(f'black_image {black_image.size}')
    #     print(f'box {box}')
        
    #     # convert box to int
    #     box = tuple(map(int, box))
        
    #     black_image.paste(cropped_region, (min_x, min_y, min_x + cropped_region.shape[1], min_y + cropped_region.shape[2]))
    
        
    #     black_image.paste(cropped_region, box)
    #     black_image = torch.from_numpy(np.array(black_image))
        
    #     # print(f"cropped_region.shape: {cropped_region.shape}")
    #     print(f'image {image.shape}')
    #     print(f'mask {mask.shape}')
    #     print(f'black_image {black_image.shape}')
    #     # print(f"cropped_region2.shape: {cropped_region2.shape}")
    #     print("--------------------------------------------")    
        
           
    #     # save the black_image for visualization
    #     black_image22 = np.squeeze(black_image, axis=2)
    #     black_image22 = black_image * 255
    #     black_image22 = black_image22.astype(np.uint8)
    #     black_image22 = Image.fromarray(black_image22)
    #     black_image22.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions2/black_image{i}.png")
        
    #     # save the mask for visualization
    #     # maskIm = np.squeeze(maskIm, axis=2)
    #     maskIm = mask * 255
    #     maskIm = maskIm.astype(np.uint8)
    #     maskIm = Image.fromarray(maskIm)
    #     maskIm.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions2/maskIm{i}.png")
        
        




    #     # # save the cropped region for visualization
    #     # # cropped_region22 = np.squeeze(cropped_region2, axis=2)
    #     # cropped_region22 = cropped_region2 * 255
    #     # cropped_region22 = cropped_region22.astype(np.uint8)
    #     # cropped_region22 = Image.fromarray(cropped_region22)
    #     # cropped_region22.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions/cropped_region2{i}.png")
    #     # # cropped_region22.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions/cropped_region2.png")

    #     # # save the cropped region for visualization
    #     # # cropped_region1 = np.squeeze(cropped_region, axis=2)
    #     # cropped_region1 = np.array(cropped_region) * 255
    #     # cropped_region1 = cropped_region1.astype(np.uint8)
    #     # cropped_region1 = Image.fromarray(cropped_region1)
    #     # # cropped_region1.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions/cropped_region{i}.png")
    #     # # cropped_region1.save(f"/mayo_atlas/home/m288756/stable-diffusion/regions/cropped_region.png")


    #     # batch = {"image": image, "mask": mask, "masked_image": masked_image}
    #     # batch = {"image": image, "mask": mask, "masked_image": cropped_region2}
    #     batch = {"image": image, "mask": mask, "masked_image": black_image}
    #     for k in batch:
    #         batch[k] = batch[k] * 2.0 - 1.0

    #     return batch
    
    
    
        # print(f"image.shape: {image.shape}")
        # print(f"mask.shape: {mask.shape}")
        # print(f"After cropped_region.shape: {cropped_region.shape}")
        # print("=====================================")