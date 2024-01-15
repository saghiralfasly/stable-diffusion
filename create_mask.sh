#!/bin/bash

python scripts/generate_masks.py \
     /mayo_atlas/home/m288756/stable-diffusion/data/images/ \
     /mayo_atlas/home/m288756/stable-diffusion/data/masks/ \
    --ext jpg \
    --max_aspect_ratio 2 \
    --num_workers 1 \
    --min_height 100 \
    --min_width 100 \
    --max_width 300 \
    --max_height 300 \

# python scripts/generate_masks.py \
#      /mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_val/ \
#      /mayo_atlas/atlas/publicDatasets/TCGA_patches/Patches1024Peyman/patches_val_masks/ \
#     --ext jpg \
#     --max_aspect_ratio 2 \
#     --num_workers 1 \
#     --min_height 100 \
#     --min_width 100 \
#     --max_width 300 \
#     --max_height 300 \