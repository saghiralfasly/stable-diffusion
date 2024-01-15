#!/usr/bin/env python3

import traceback
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
import glob
import os

class MakeManyMasksWrapper:
    def __init__(self, variants_n=2):
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self._generate_random_mask(img) for _ in range(self.variants_n)]

    @staticmethod
    def _generate_random_mask(img, min_width=30, max_width=None, min_height=30, max_height=None, max_aspect_ratio=None):
        height, width = img.shape[1], img.shape[2]

        # Randomly select width and height based on provided arguments
        width_rect = np.random.randint(min_width, max_width + 1) if max_width else np.random.randint(min_width, width + 1)
        height_rect = np.random.randint(min_height, max_height + 1) if max_height else np.random.randint(min_height, height + 1)

        # Calculate remaining offset for center coordinates
        offset_width = width - width_rect
        offset_height = height - height_rect

        # Randomly select center coordinates within the remaining offset
        center_y = np.random.randint(offset_height + 1)
        center_x = np.random.randint(offset_width + 1)

        # Calculate top-left and bottom-right coordinates
        top_left_y = center_y
        top_left_x = center_x
        bottom_right_y = center_y + height_rect
        bottom_right_x = center_x + width_rect

        random_mask = np.zeros((height, width), dtype='float32')
        random_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1.0

        return random_mask


def process_images(src_images, indir, outdir, min_height=30, min_width=30, max_width=None, max_height=None, max_aspect_ratio=None):
    mask_generator = MakeManyMasksWrapper(variants_n=2)
    max_tamper_area = 1

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            if min(image.size) < 1024:
                continue
            else:
                factor = 1024 / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue
                filtered_image_mask_pairs.append((image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), 1),
                                            replace=False)

            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + '.jpg')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Error processing {infile}: {ex}\n{traceback.format_exc()}')

def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'

    os.makedirs(args.outdir, exist_ok=True)

    in_files = [f for f in glob.glob(os.path.join(args.indir, '**', f'*.{args.ext}'), recursive=True)]
    if args.num_workers == 0:
        process_images(in_files, args.indir, args.outdir, args.min_height, args.min_width, args.max_width, args.max_height, args.max_aspect_ratio)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // args.num_workers + (1 if in_files_n % args.num_workers > 0 else 0)
        Parallel(n_jobs=args.num_workers)(
            delayed(process_images)(in_files[start:start+chunk_size], args.indir, args.outdir, args.min_height, args.min_width, args.max_width, args.max_height, args.max_aspect_ratio)
            for start in range(0, len(in_files), chunk_size)
        )

if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')
    aparser.add_argument('--num_workers', type=int, default=0, help='Number of processes to use')
    aparser.add_argument('--ext', type=str, default='jpg', help='Input image extension')
    aparser.add_argument('--min_height', type=int, default=30, help='Minimum height of the generated masks')
    aparser.add_argument('--min_width', type=int, default=30, help='Minimum width of the generated masks')
    aparser.add_argument('--max_width', type=int, default=None, help='Maximum width of the generated masks (default is unlimited)')
    aparser.add_argument('--max_height', type=int, default=None, help='Maximum height of the generated masks (default is unlimited)')
    aparser.add_argument('--max_aspect_ratio', type=float, default=None, help='Maximum aspect ratio between width and height for generated masks')

    main(aparser.parse_args())
