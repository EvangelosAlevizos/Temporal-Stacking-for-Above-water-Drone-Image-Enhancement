#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal Stacking for Above-water Drone Image Enhancement

Aligns drone images and applies temporal filtering (min, median, percentile)
to reduce glint, and bottom caustics.

Author: Evangelos Alevizos
Version: 1.0 (24.04.2026)
"""

import os
import cv2
import gc
import argparse
import subprocess
import numpy as np
import imageio
from collections import defaultdict


# -------------------------------
# ARGUMENTS
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal stacking of drone images"
    )

    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--chunk", type=int, default=11, help="Images per chunk")
    parser.add_argument("--percentile", type=float, default=1,
                        help="0=min, 1=median, otherwise percentile")
    parser.add_argument("--downscale", type=float, default=0.25,
                        help="Downscale factor for alignment")
    parser.add_argument("--no_metadata", action="store_true",
                        help="Disable metadata copy")

    return parser.parse_args()


# -------------------------------
# IMAGE UTILITIES
# -------------------------------
def read_image(path):
    img = imageio.imread(path)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    return img.astype(np.float32) / 255.0


# -------------------------------
# ALIGNMENT
# -------------------------------
def align_image_ecc_affine_downsample(ref, mov, n_iter=1000, scale=0.25):
    h, w = ref.shape[:2]

    ref_gray = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor((mov * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale).astype(np.float32)
    mov_small = cv2.resize(mov_gray, None, fx=scale, fy=scale).astype(np.float32)

    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        1e-6
    )

    try:
        _, warp = cv2.findTransformECC(
            ref_small, mov_small,
            warp, cv2.MOTION_AFFINE,
            criteria, None, 5
        )
    except cv2.error:
        return mov, np.ones((h, w), dtype=bool)

    warp[:, 2] /= scale

    aligned = cv2.warpAffine(
        mov, warp, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    valid_mask = cv2.warpAffine(
        np.ones((h, w), np.uint8),
        warp, (w, h),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
    ).astype(bool)

    return aligned, valid_mask


# -------------------------------
# MAIN PROCESSING
# -------------------------------
def process_images(args):

    input_dir = args.input
    output_dir = args.output
    chunk_size = args.chunk
    percentile = args.percentile
    downscale = args.downscale

    os.makedirs(output_dir, exist_ok=True)

    # Collect images
    image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        raise ValueError("No JPG images found in input directory")

    print(f"Found {len(image_files)} images")

    # Chunking
    chunks = defaultdict(list)
    for idx in range(len(image_files)):
        chunks[idx // chunk_size].append(idx)

    print(f"Detected {len(chunks)} chunks")

    # Processing loop
    for c, idxs in chunks.items():
        print(f"\nProcessing chunk {c+1}/{len(chunks)}")

        ref_img = read_image(image_files[idxs[0]])
        aligned_imgs = [ref_img]
        valid_masks = [np.ones(ref_img.shape[:2], dtype=bool)]

        for i in idxs[1:]:
            mov_img = read_image(image_files[i])
            aligned, mask = align_image_ecc_affine_downsample(
                ref_img, mov_img, scale=downscale
            )

            aligned_imgs.append(aligned)
            valid_masks.append(mask)

            del mov_img
            gc.collect()

        # Crop valid area
        valid = np.logical_and.reduce(valid_masks)
        ys, xs = np.where(valid)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        aligned_imgs = [
            img[y0:y1+1, x0:x1+1] for img in aligned_imgs
        ]

        # Temporal filtering
        stack = np.stack(aligned_imgs, axis=0)

        if percentile == 0:
            out_img = np.min(stack, axis=0)
        elif percentile == 1:
            out_img = np.median(stack, axis=0)
        else:
            out_img = np.percentile(stack, percentile, axis=0)

        out_uint8 = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)

        out_name = f"filt_{os.path.basename(image_files[idxs[0]])}"
        out_path = os.path.join(output_dir, out_name)

        imageio.imwrite(out_path, out_uint8, quality=95, subsampling=0)

        # Metadata copy
        if not args.no_metadata:
            try:
                subprocess.run([
                    "exiftool",
                    "-TagsFromFile", image_files[idxs[0]],
                    "-all:all",
                    "-overwrite_original",
                    out_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("Warning: ExifTool not found, skipping metadata copy")

        # Cleanup
        del aligned_imgs, valid_masks, stack, out_img, out_uint8, ref_img
        gc.collect()

    print("\nAll chunks processed successfully!")


# -------------------------------
# ENTRY POINT
# -------------------------------
def main():
    args = parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
