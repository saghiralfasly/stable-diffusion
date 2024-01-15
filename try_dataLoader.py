from ldm.data import inpainting_image, inpainting_image_default
from torch.utils.data import random_split, DataLoader

img_size = 512
path="/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_val/filelistAll.txt"

dataSet = inpainting_image.InpaintingTrain(img_size,path)
# dataSet = inpainting_image_default.InpaintingTrain(img_size,path)
print(f'Training LDM with {len(dataSet)} images')

data = DataLoader(
        dataSet,
        batch_size=1,
        shuffle=True,
        num_workers=1,
)

for i, batch in enumerate(data):
    image = batch['image']
    mask = batch['mask']
    masked_image = batch['masked_image']
    print(f"image.shape: {image.shape}")
    print(f"mask.shape: {mask.shape}")
    print(f"masked_image.shape: {masked_image.shape}")
    print("--------------------------------------------")    
    if i == 2:
        break