#!/usr/bin/env python3

import glob
import os
import traceback

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

class MakeManyMasksWrapper:
    def __init__(self, variants_n=2):
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self._generate_random_mask(img) for _ in range(self.variants_n)]

    @staticmethod
    def _generate_random_mask(img, min_size=30):
        # Assuming img is a 3D numpy array representing an RGB image
        height, width = img.shape[1], img.shape[2]

        # Set a maximum aspect ratio to avoid very thin rectangles
        max_aspect_ratio = 2.0

        while True:
            # Generate random width and height with controlled aspect ratio
            aspect_ratio = np.random.uniform(1.0, max_aspect_ratio)
            height_rect = int(np.sqrt(np.random.uniform(0.5, 1.0) * height**2 / aspect_ratio))
            width_rect = int(aspect_ratio * height_rect)

            # Ensure the generated dimensions are at least 30 pixels
            if height_rect >= min_size and width_rect >= min_size:
                break

        # Generate random coordinates for the top-left corner
        top_left_y = np.random.randint(0, height)
        top_left_x = np.random.randint(0, width)

        # Calculate the bottom-right coordinates
        bottom_right_y = min(top_left_y + height_rect, height)
        bottom_right_x = min(top_left_x + width_rect, width)

        # Create a mask with zeros and set the region inside the rectangle to ones
        random_mask = np.zeros((height, width), dtype='float32')
        random_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1.0

        return random_mask




def process_images(src_images, indir, outdir):
    mask_generator = MakeManyMasksWrapper(variants_n=2)
    max_tamper_area = 1

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < 512:  # Example threshold, adjust as needed
                continue
            else:
                factor = 512 / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), 1),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                # cur_basename = mask_basename + f'_crop{i:03d}'
                cur_basename = mask_basename # + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + '.jpg') #mode='L').save(cur_basename + f'_mask.png') # mode='L').save(cur_basename + f'_mask{i:03d}.png') # mode='L').save(cur_basename + f'_mask{i:03d}.png')
                # cur_image.save(cur_basename + '.jpg')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')

def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True))
    if args.n_jobs == 0:
        process_images(in_files, args.indir, args.outdir)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.n_jobs + (1 if in_files_n % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir)
            for start in range(0, len(in_files), chunk_size)
        )

if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--num_workers', type=int, default=0, help='How many processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')

    main(aparser.parse_args())
