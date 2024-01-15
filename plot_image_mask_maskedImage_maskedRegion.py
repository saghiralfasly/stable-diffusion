import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_flist, size):
        self.image_flist = image_flist
        self.size = size

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
    
    def process_image(self, i):
        image = np.array(Image.open(self.image_flist[i]).convert("RGB"))
        image = cv2.resize(image, (self.size, self.size))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        mask = self.generate_stroke_mask([self.size, self.size])
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = (1 - mask) * image

        masked_image_patch = image * mask
        masked_image_patch = masked_image_patch + (1 - mask)
        masked_image_patch = masked_image_patch * 2.0 - 1.0

        batch = {"image": image, "mask": mask, "masked_image": masked_image}
        for k in batch:
            batch[k] = batch[k] * 2.0 - 1.0

        return batch

def main():
    # Provide the path to your input image
    input_image_path = "/mayo_atlas/home/m288756/stable-diffusion/data/oneImage/TCGA-22-A5C4-01Z-00-DX1.54058689-5CA5-4F92-B18A-86208C24C87D_18432_47104.jpg"

    # Example usage
    image_processor = ImageProcessor(image_flist=[input_image_path], size=512)
    result_batch = image_processor.process_image(0)

    # Access the processed data
    processed_image = result_batch["image"]
    processed_mask = result_batch["mask"]
    processed_masked_image = result_batch["masked_image"]

    # Plot the images
    plt.figure(figsize=(10, 4))

    plt.subplot(131)
    plt.imshow(np.transpose(processed_image.numpy(), (1, 2, 0)))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(processed_mask.numpy(), cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(np.transpose(processed_masked_image.numpy(), (1, 2, 0)))
    plt.title("Masked Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
