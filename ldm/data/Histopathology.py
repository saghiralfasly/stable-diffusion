import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt

from scipy import ndimage

def get_square_crop(mask):
    ''' this function receives a mask and returns the dame mask with just square crop of the largest region'''
    # loop over the mask with a sliding window started with size 200x200 if you don't find a region with all 1s then reduce the size by 10 (will be 190x190). Overall, i want a patch of size between 200x200 and 32x32. then return a mask that contains only the this new patch
    for i in range(200, 32, -10):
        for j in range(0, mask.shape[0] - i, 10):
            for k in range(0, mask.shape[1] - i, 10):
                # print(i, j, k)
                # print(mask[j:j+i, k:k+i])
                if np.all(mask[j:j+i, k:k+i] == 1):
                    # print(i,j,k)
                    # print("-----------")
                    # if you find a patch of size larger than 100x100 then do random center crop of size between ixi and 64x64
                    if i > 100:
                        rand_new_i = np.random.randint(64, i)
                        rand_new_j = np.random.randint(64, i)
                        
                        new_mask = np.zeros_like(mask)
                        new_mask[j:j+rand_new_i, k:k+rand_new_j] = mask[j:j+rand_new_i, k:k+rand_new_j]
                        # new_mask[j:j+i, k:k+i] = mask[j:j+i, k:k+i]
                        # print(rand_new_i, rand_new_j)
                        # print("---^^^^^^^^^^^^^^^---")
                    else:
                        new_mask = np.zeros_like(mask)
                        new_mask[j:j+i, k:k+i] = mask[j:j+i, k:k+i]
                    return new_mask

def get_largest_region_crop(mask):
    # Label connected components in the binary mask
    labeled_mask, num_labels = ndimage.label(mask)

    # Calculate properties of each labeled region
    regions = ndimage.find_objects(labeled_mask)

    # Find the largest region
    largest_region_index = max(range(1, num_labels + 1), key=lambda i: np.sum(labeled_mask == i))

    # Extract the region corresponding to the largest label
    largest_region_slice = regions[largest_region_index - 1]

    # Handle the case where the region is not found
    if largest_region_slice is None:
        return None

    start_row, start_col = largest_region_slice[0].start, largest_region_slice[1].start
    end_row, end_col = largest_region_slice[0].stop, largest_region_slice[1].stop

    # Extract the crop from the largest region
    largest_region_crop = mask[start_row:end_row, start_col:end_col]

    return {
        'coordinates': (start_row, start_col, end_row, end_col),
        'height': end_row - start_row,
        'width': end_col - start_col,
        'largest_region_crop': largest_region_crop
    }
    
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
        
        # print(segmentation.tolist())
        
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
                #  n_labels=3, shift_segmentation=False,
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
        # print(segmentation.tolist())
        segmentation = segmentation.astype(np.uint8)
        # convert all values >= self.n_labels to self.n_labels-1 in order to avoid errors in onehot
        # segmentation[segmentation >= self.n_labels] = self.n_labels-1
        
        # print(segmentation.tolist())
        
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
        
        # tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        # not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1)     
        
           
        # get updated_onehot where it contains square crops only from the onehot segmentation. the crops are 30% of the onehot segmentation
        onehop1 = get_square_crop(onehot[:,:,0])
        onehop2 = get_square_crop(onehot[:,:,1])
        
        if onehop1 is None:
            new_tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        else:
            new_tumor_crop = example["image"] * np.expand_dims(onehop1, axis=-1)
            
        if onehop2 is None:
            new_not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1) 
        else:
            new_not_tumor_crop = example["image"]  * np.expand_dims(onehop2, axis=-1) 
            
        # print(new_tumor_crop[:,:,0].tolist()) 
        # print(new_not_tumor_crop.shape)

        # now I want to create segment_regions which contains last dimension as follows [:,:,-l]: 
        #        the first onehot segmentation with index 0, then the tumor_crop, 
        #        then the second onehot segmentation with index 1, then the not_tumor_crop
        example["segmentation"] = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), new_tumor_crop, np.expand_dims(onehot[:,:,1], axis=-1), new_not_tumor_crop), axis=-1) 
        # example["segmentation"] = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), tumor_crop, np.expand_dims(onehot[:,:,1], axis=-1), not_tumor_crop), axis=-1) 
        # print(segment_regions.shape)     
        
        # # Save images in two rows using plt
        # plt.figure(figsize=(16, 4))

        # tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        # not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1)  
        
        # # First row: Original Image, Generated Mask, Masked Image
        # img = example["image"]
        # plt.subplot(1, 6, 1)
        # plt.imshow(((img + 1) * 127.5).astype(np.uint8))
        # plt.title('Original Image')
        # plt.axis('off')

        
        # plt.subplot(1, 6, 2)
        # plt.imshow(segmentation)
        # plt.title('Segmentation Map')
        # plt.axis('off')

        # plt.subplot(1, 6, 3)
        # plt.imshow(((tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Normal Prompt')
        # plt.axis('off')

        # plt.subplot(1, 6, 4)
        # plt.imshow(((not_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Tumor Prompt')
        # plt.axis('off')
        
        # plt.subplot(1, 6, 5)
        # plt.imshow(((new_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Normal Prompt')
        # plt.axis('off')

        # plt.subplot(1, 6, 6)
        # plt.imshow(((new_not_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Tumor Prompt')
        # plt.axis('off')

         
        # plt.show()

        # plt.savefig(f"outputs/semantic/prompts_updated/sample_{i}_before.png")
        # plt.close()
        
        # print(example["file_path_"])
            
        return example
    
# class HistoTrain(SegmentationBaseHisto):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
#                          segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
#                          size=size, random_crop=random_crop, interpolation=interpolation)
        
# class HistoVal(SegmentationBaseHisto):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
#                          segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
#                          size=size, random_crop=random_crop, interpolation=interpolation)

        
# class HistoTrainPrompt(SegmentationBaseHistoPrompt):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
#                          segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/masks_all/",
#                          size=size, random_crop=random_crop, interpolation=interpolation)
        
# class HistoValPrompt(SegmentationBaseHistoPrompt):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
#                          segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks_all/",
#                          size=size, random_crop=random_crop, interpolation=interpolation)
        
        
# class Examples(SegmentationBaseHisto):
#     def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
#         super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks/",
#                          segmentation_root="/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/testing/masks/",
#                          size=size, random_crop=random_crop, interpolation=interpolation)




class SegmentationBaseHistoPromptPanda(Dataset):
    def __init__(self,
                 data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=3, shift_segmentation=False,
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
        # print(segmentation.tolist())
        segmentation = segmentation.astype(np.uint8)
        # convert all values >= self.n_labels to self.n_labels-1 in order to avoid errors in onehot
        segmentation[segmentation >= self.n_labels] = self.n_labels-1
        
        # print(segmentation.tolist())
        
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
        
        # tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        # not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1)     
        
           
        # get updated_onehot where it contains square crops only from the onehot segmentation. the crops are 30% of the onehot segmentation
        onehop1 = get_square_crop(onehot[:,:,1])
        onehop2 = get_square_crop(onehot[:,:,2])
        
        if onehop1 is None:
            new_tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        else:
            new_tumor_crop = example["image"] * np.expand_dims(onehop1, axis=-1)
            
        if onehop2 is None:
            new_not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1) 
        else:
            new_not_tumor_crop = example["image"]  * np.expand_dims(onehop2, axis=-1) 
            
        # print(new_tumor_crop[:,:,0].tolist()) 
        # print(new_not_tumor_crop.shape)

        # now I want to create segment_regions which contains last dimension as follows [:,:,-l]: 
        #        the first onehot segmentation with index 0, then the tumor_crop, 
        #        then the second onehot segmentation with index 1, then the not_tumor_crop
        example["segmentation"] = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), np.expand_dims(onehot[:,:,1], axis=-1), new_tumor_crop, np.expand_dims(onehot[:,:,2], axis=-1), new_not_tumor_crop), axis=-1) 
        # example["segmentation"] = np.concatenate((np.expand_dims(onehot[:,:,0], axis=-1), tumor_crop, np.expand_dims(onehot[:,:,1], axis=-1), not_tumor_crop), axis=-1) 
        # print(segment_regions.shape)     
        
        # # Save images in two rows using plt
        # plt.figure(figsize=(16, 4))

        # tumor_crop = example["image"] * np.expand_dims(onehot[:,:,0], axis=-1)
        # not_tumor_crop = example["image"]  * np.expand_dims(onehot[:,:,1], axis=-1)  
        
        # # First row: Original Image, Generated Mask, Masked Image
        # img = example["image"]
        # plt.subplot(1, 6, 1)
        # plt.imshow(((img + 1) * 127.5).astype(np.uint8))
        # plt.title('Original Image')
        # plt.axis('off')

        
        # plt.subplot(1, 6, 2)
        # plt.imshow(segmentation)
        # plt.title('Segmentation Map')
        # plt.axis('off')

        # plt.subplot(1, 6, 3)
        # plt.imshow(((tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Normal Prompt')
        # plt.axis('off')

        # plt.subplot(1, 6, 4)
        # plt.imshow(((not_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Tumor Prompt')
        # plt.axis('off')
        
        # plt.subplot(1, 6, 5)
        # plt.imshow(((new_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Normal Prompt')
        # plt.axis('off')

        # plt.subplot(1, 6, 6)
        # plt.imshow(((new_not_tumor_crop + 1) * 127.5).astype(np.uint8))
        # plt.title('Tumor Prompt')
        # plt.axis('off')

         
        # plt.show()

        # plt.savefig(f"outputs/semantic/prompts_updated/sample_{i}_before.png")
        # plt.close()
        
        # print(example["file_path_"])
            
        return example

class HistoTrainPromptPanda(SegmentationBaseHistoPromptPanda):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/PANDA/masks/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/PANDA/masks/",
                         size=size, random_crop=random_crop, interpolation=interpolation)
        
class HistoValPromptPanda(SegmentationBaseHistoPromptPanda):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_root="/mayo_atlas/atlas/publicDatasets/PANDA/masks/",
                         segmentation_root="/mayo_atlas/atlas/publicDatasets/PANDA/masks/",
                         size=size, random_crop=random_crop, interpolation=interpolation)