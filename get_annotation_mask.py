from xml.dom import minidom

import xml.etree.ElementTree as ET
import cv2

import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

import openslide
import numpy as np

            
def get_annotation_mask(annotation_path, dimensions, down_sample):
    mask_shape = (
        round(dimensions[1] * down_sample),
        round(dimensions[0] * down_sample),
    )
    # Initialize the mask as an array of zeros
    # mask2 = np.zeros(mask_shape, dtype=np.uint8)
    mask = np.full(mask_shape, False)

    annotations = minidom.parse(annotation_path).getElementsByTagName("Annotation")
    for i, annotation in enumerate(annotations):
        # if i == 0:
            region_coords = []
            for j, coords in enumerate(annotation.getElementsByTagName("Coordinate")):
                # if j < 10:
                #     print(float(coords.attributes["Y"].value))
                #     print(float(coords.attributes["X"].value))

                region_coords.append((float(coords.attributes["Y"].value) * down_sample, float(coords.attributes["X"].value) * down_sample))
            mask = mask ^ draw.polygon2mask(mask_shape, region_coords)
            # count the number of ones in the mask
            print(np.count_nonzero(mask))
            print(np.nonzero(mask))
    # Convert region_coords to a format that fillPoly can use
    # region_coords = np.array(region_coords, dtype=np.int32)
    # Reshape region_coords to a 3D array
    # region_coords = region_coords.reshape((-1, 1, 2))

    # Use fillPoly to update the mask
    # cv2.fillPoly(mask2, [region_coords], 1)
    return mask#, mask2



def main():
    # Path to your image and XML file
    # image_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/tumor/tumor_081.tif"
    # xml_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/lesion_annotations/tumor_081.xml"

    # image_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/tumor/tumor_026.tif"
    # xml_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/lesion_annotations/tumor_026.xml"
    
    image_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/tumor/tumor_064.tif"
    xml_path = "/mayo_atlas/atlas/publicDatasets/CAMELYON_16e/training/lesion_annotations/tumor_064.xml"
    
    # Open the slide image
    slide = openslide.OpenSlide(image_path)

    # Get the dimensions of the image
    dimensions = slide.dimensions
    
    print(dimensions)
    
    # show the image thumbnail
    slide_thumbnail = slide.get_thumbnail((2000, 2000))
    # resize the thumbnail to the no more than 2000x2000 but with the same aspect ratio
    # slide_thumbnail = slide_thumbnail.resize((2000, 2000))
    # save it to the disk
    slide_thumbnail.save('thumbnail.png')

    # # Define the downsample factor
    # down_sample = 0.2  # Adjust as needed
    down_sample = 0.2  # Adjust as needed
    
    # parse_annotation_xml(xml_path)

    # # # Get the annotation mask
    mask = get_annotation_mask(xml_path, dimensions, down_sample)
    
    print(mask.shape)
    mask = mask * 255
    


    # Ensure mask is a NumPy array
    mask = np.array(mask, dtype=np.uint8)
    # plt.imshow(mask)
    # plt.savefig('mask.png')
    # print(mask)
    # find the number of ones in the mask
    print(np.count_nonzero(mask))
    mask_dim = mask.shape
    
    # # print the value of the non zero elements
    # print(np.nonzero(mask))
    
    # resize the mask to the no more than 2000x2000 but with the same aspect ratio
    # mask2 = cv2.resize(mask, (int(mask.shape[0]* 0.1), int(mask.shape[1]* 0.1)), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (int(mask_dim[1] * 0.2), int(mask_dim[0] * 0.2)))
    # print(mask)

    # Save the mask
    cv2.imwrite('mask.png', mask)


if __name__ == "__main__":
    main()