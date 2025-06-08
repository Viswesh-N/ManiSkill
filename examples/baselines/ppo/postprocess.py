#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def parse_args():
    p = argparse.ArgumentParser(
        description="Mask out objects, edit background via diffusion, and paste objects back"
    )
    p.add_argument(
        "--input-dir", type=str, required=True,
        help="Root directory of collected data (contains metadata/ and images/)"
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save masked, edited images and metadata"
    )
    p.add_argument(
        "--prompt", type=str,
        default="Very slightly change the background but keep the table and overall strucute intact",
        help="Diffusion prompt for background editing"
    )
    p.add_argument(
        "--inpaint-method", type=str,
        choices=["black", "white", "blur", "edge_extend"],
        default="white",
        help="Method to mask out objects before editing"
    )
    p.add_argument(
        "--num-steps", type=int, default=25,
        help="Number of diffusion inference steps"
    )
    p.add_argument(
        "--guidance-scale", type=float, default=2.5,
        help="Image guidance scale for diffusion"
    )
    return p.parse_args()


def create_object_mask(seg_arr, obj_ids):
    mask = np.zeros(seg_arr.shape, dtype=bool)
    for oid in obj_ids:
        mask |= (seg_arr == oid)
    return mask


def enhance_prompt(base_prompt, has_arm, has_cube):
    if has_arm and has_cube:
        return base_prompt + " Fill in the spaces where the arm and cube were removed."
    if has_arm:
        return base_prompt + " Fill in the space where the robotic arm was removed."
    if has_cube:
        return base_prompt + " Fill in the space where the cube was removed."
    return base_prompt


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Setup output dirs
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masked_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    # Load metadata
    col_meta_path = os.path.join(input_dir, "metadata/collection_metadata.json")
    seg_map_path = os.path.join(input_dir, "metadata/segmentation_id_map.json")
    with open(col_meta_path) as f:
        orig_meta = json.load(f)
    with open(seg_map_path) as f:
        seg_id_map = json.load(f)

    # Setup diffusion pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Identify arm and cube IDs
    arm_ids, cube_ids = [], []
    for sid_str, info in seg_id_map.items():
        sid = int(sid_str)
        if sid == 0:
            continue
        nm = info["name"].lower()
        cl = info["class_name"].lower()
        if any(x in nm for x in ("arm","gripper")) or "link" in cl:
            arm_ids.append(sid)
        elif "cube" in nm or "object" in nm:
            cube_ids.append(sid)
    all_ids = arm_ids + cube_ids

    # Enhance prompt
    enhanced = enhance_prompt(args.prompt, bool(arm_ids), bool(cube_ids))

    # Process each image
    processed = {
        "original_images": [],
        "segmentation_masks": [],
        "masked_images": [],
        "edited_images": [],
        "actions": [],
        "rewards": [],
        "terminations": [],
        "truncations": [],
        "arm_ids": arm_ids,
        "cube_ids": cube_ids,
        "prompt": args.prompt,
        "enhanced_prompt": enhanced,
    }

    for img_rel, seg_rel, act, rew, term, trunc in tqdm(
        zip(orig_meta["images"], orig_meta["segmentation_masks"], orig_meta["actions"],
            orig_meta["rewards"], orig_meta["terminations"], orig_meta["truncations"]),
        total=len(orig_meta["images"]), desc="Editing"
    ):
        # Load image and segmentation
        img = Image.open(os.path.join(input_dir, img_rel))
        img = ImageOps.exif_transpose(img).convert("RGB")
        orig_np = np.array(img)
        seg_np = np.array(Image.open(os.path.join(input_dir, seg_rel)))

        # Create mask
        mask = create_object_mask(seg_np, all_ids)

        # Mask out objects
        masked = orig_np.copy()
        if args.inpaint_method == "black":
            masked[mask] = 0
        elif args.inpaint_method == "white":
            masked[mask] = 255
        elif args.inpaint_method == "blur":
            from scipy import ndimage
            for c in range(3):
                _, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
                masked[...,c][mask] = orig_np[...,c][idx[0][mask], idx[1][mask]]
        else:  # edge_extend
            masked[mask] = orig_np[mask]

        # Save masked image
        fname = os.path.basename(img_rel)
        Image.fromarray(masked).save(os.path.join(output_dir, "masked_images", fname))

        # Run diffusion edit
        edited = pipe(
            image=Image.fromarray(masked),
            prompt=enhanced,
            num_inference_steps=args.num_steps,
            image_guidance_scale=args.guidance_scale
        ).images[0]

        # Paste objects back
        edt_np = np.array(edited)
        edt_np[mask] = orig_np[mask]
        Image.fromarray(edt_np).save(os.path.join(output_dir, "images", fname))

        # Record
        processed["original_images"].append(img_rel)
        processed["segmentation_masks"].append(seg_rel)
        processed["actions"].append(act)
        processed["rewards"].append(rew)
        processed["terminations"].append(term)
        processed["truncations"].append(trunc)
        processed["masked_images"].append(f"masked_images/{fname}")
        processed["edited_images"].append(f"images/{fname}")

    # Write metadata
    out_meta = os.path.join(output_dir, "metadata/processed_metadata.json")
    with open(out_meta, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Done. Edited images in {output_dir}/images and metadata in {out_meta}")

if __name__ == "__main__":
    main()
