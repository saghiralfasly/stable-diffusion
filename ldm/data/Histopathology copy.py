import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt

class SegmentationBaseHisto(Dataset):
    def __init__(self,
                 data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                #  n_labels=182, shift_segmentation=False, # ***********************************
                 n_labels=2, shift_segmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        # self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        # read the image paths from the folder with the images of png format
        self.image_paths = glob.glob(os.path.join(self.data_root, "*.png"))
        # print(self.image_paths)
        # self.image_paths = self
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [l for l in self.image_paths],
            "segmentation_path_": [l.replace(".png", ".npy") for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        # segmentation = Image.open(example["segmentation_path_"]) # *******************************************************
        
        # print(example["segmentation_path_"])
        segmentation = np.load(example["segmentation_path_"])
        segmentation = segmentation.astype(np.uint8)
        
        # assert segmentation.mode == "L", segmentation.mode
        # segmentation = np.array(segmentation).astype(np.uint8)
        
        # if self.shift_segmentation:
        #     # used to support segmentations containing unlabeled==255 label
        #     segmentation = segmentation+1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        # segmentation = processed["mask"]
        # print(segmentation)
        # print(segmentation.shape)
        # onehot = np.eye(self.n_labels)[segmentation]
        # print(onehot.shape)
        # # convert np to list
        # print(onehot[0].tolist())
        # print("00000000000000000000000000000000")
        # print(onehot[1].tolist())
        # print(onehot.shape)
        
        # mask = np.load("/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks/tumor_057/tumor_057_14320_156446.npy")
        segmentation = processed["mask"]
        segmentation = segmentation.astype(np.uint8)
        onehot = np.eye(self.n_labels)[segmentation]
        # print(onehot.shape)
        # # print(onehot.tolist())
        
        # onehot = onehot.transpose(2, 0, 1)
        # # print(onehot.tolist())
        # print(onehot.shape)
        # print("_________________________________________________")
        


        example["segmentation"] = onehot
        return example


class SegmentationBaseHistoPrompt(Dataset):
    def __init__(self,
                 data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                #  n_labels=182, shift_segmentation=False, # ***********************************
                 n_labels=2, shift_segmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        # self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        # read the image paths from the folder with the images of png format
        self.image_paths = glob.glob(os.path.join(self.data_root, "*.png"))
        # print(self.image_paths)
        # self.image_paths = self
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [l for l in self.image_paths],
            "segmentation_path_": [l.replace(".png", ".npy") for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        # print(example["segmentation_path_"])
        segmentation = np.load(example["segmentation_path_"])
        segmentation = segmentation.astype(np.uint8)
        
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)

        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]

        tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1)
        
        
        # Get the shape of the crops  tumor_crop
        shape = tumor_crop.shape
        # Generate a random mask size
        mask_size = np.random.randint(50, 150, size=2)
        # Create a random binary mask of the same size as the region to mask
        mask = np.random.choice([0, 1], size=mask_size)
        # Choose a random position for the top-left corner of the mask
        x = np.random.randint(0, shape[1] - mask_size[1] + 1)
        y = np.random.randint(0, shape[0] - mask_size[0] + 1)
        # Add an extra dimension to the mask
        mask_3d = np.expand_dims(mask, axis=-1)
        # Apply the mask to the region in the crops
        tumor_crop[y:y+mask_size[0], x:x+mask_size[1]] *= mask_3d

        
        
        
        # Get the shape of the crops  not_tumor_crop
        shape = not_tumor_crop.shape
        # Generate a random mask size
        mask_size = np.random.randint(100, 150, size=2)
        # Create a random binary mask of the same size as the region to mask
        mask = np.random.choice([0, 1], size=mask_size)
        # Choose a random position for the top-left corner of the mask
        x = np.random.randint(0, shape[1] - mask_size[1] + 1)
        y = np.random.randint(0, shape[0] - mask_size[0] + 1)
        # Add an extra dimension to the mask
        mask_3d = np.expand_dims(mask, axis=-1)
        # Apply the mask to the region in the crops
        not_tumor_crop[y:y+mask_size[0], x:x+mask_size[1]] *= mask_3d

        # now I want to create segment_regions which contains last dimension as follows [:,:,-l]: 
        #        the first onehot segmentation with index 0, then the tumor_crop, 
        #        then the second onehot segmentation with index 1, then the not_tumor_crop
        example["segmentation"] = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), tumor_crop, np.expand_dims(onehot[:,:,1], axis=-1), not_tumor_crop), axis=-1) 
        # print(segment_regions.shape)

        # example["segmentation"] = onehot
        
        
        # Save images in two rows using plt
        plt.figure(figsize=(12, 8))

        # First row: Original Image, Generated Mask, Masked Image
        img = example["image"]
        print(img.shape)
        print(type(img))
        print(img[0])
        plt.subplot(1, 5, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')

        print(segmentation.shape)
        print(type(segmentation))
        
        plt.subplot(1, 5, 2)
        plt.imshow(segmentation)
        plt.title('Segmentation Map')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.imshow(tumor_crop)
        plt.title('Tumor Prompt')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        plt.imshow(not_tumor_crop)
        plt.title('Normal Prompt')
        plt.axis('off')

        # plt.subplot(1, 5, 5)
        # plt.imshow(output_PIL)
        # plt.title('Generated Image')
        # plt.axis('off')
        
        plt.show()

        plt.savefig(f"outputs/semantic/prompts/sample_.png")
        plt.close()
        
        print(example["file_path_"])
            
        return example
    
class HistoTrain(SegmentationBaseHisto):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
                         size=size, random_crop=random_crop, interpolation=interpolation)
        
class HistoVal(SegmentationBaseHisto):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
                         size=size, random_crop=random_crop, interpolation=interpolation)

        
class HistoTrainPrompt(SegmentationBaseHistoPrompt):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
                         size=size, random_crop=random_crop, interpolation=interpolation)
        
class HistoValPrompt(SegmentationBaseHistoPrompt):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
                         size=size, random_crop=random_crop, interpolation=interpolation)
        
        
class Examples(SegmentationBaseHisto):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks/",
                         size=size, random_crop=random_crop, interpolation=interpolation)