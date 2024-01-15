import os

def create_txt_file(folder_path, output_file):
    with open(output_file, 'w') as txt_file:
        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is a .jpg image
            if filename.endswith(".jpg"):
                # Get the full path of the image file
                image_path = os.path.join(folder_path, filename)
                # Write the path to the text file
                txt_file.write(image_path + '\n')

# Replace 'path_to_images_folder' with the actual path to your folder
# images_folder = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_val/'
# images_folder = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_1_million/'
images_folder = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches/'

# Replace 'output.txt' with the desired name for the output text file
# output_text_file = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_val/filelistAll.txt'
# output_text_file = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_1_million/filelistAll.txt'
output_text_file = '/mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches/filelistAll.txt'
# Call the function to create the text file
create_txt_file(images_folder, output_text_file)
