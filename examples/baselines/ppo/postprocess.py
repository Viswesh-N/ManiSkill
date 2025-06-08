#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch
import torch._dynamo as dynamo
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def parse_args():
    p = argparse.ArgumentParser(
        description="Mask out objects, edit background via diffusion, and paste objects back with optimizations"
    )
    p.add_argument(
        "--input-dir", type=str, default="data",
        help="Root directory of collected data (contains metadata/)"
    )
    p.add_argument(
        "--output-dir", type=str, default="/content/outputs",
        help="Directory to save masked, edited images and metadata"
    )
    p.add_argument(
        "--prompt", type=str,
        default="Very slightly change the background but keep the table and overall structure intact.",
        help="Diffusion prompt for background editing"
    )
    p.add_argument(
        "--inpaint-method", type=str, choices=["black", "white", "blur", "edge_extend"],
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
    p.add_argument(
        "--division-id", type=int, default=1,
        help="Which division of the dataset to process (1-based)"
    )
    p.add_argument(
        "--num-divisions", type=int, default=6,
        help="Total number of equal divisions"
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
    inp = args.input_dir
    out = args.output_dir

    # Prepare output dirs
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masked_images"), exist_ok=True)
    os.makedirs(os.path.join(out, "metadata"), exist_ok=True)

    # Load metadata with fallback
    meta_dir = os.path.join(inp, "metadata")
    col_meta_path = os.path.join(meta_dir, "collection_metadata.json")
    if not os.path.exists(col_meta_path):
        alt = os.path.join(meta_dir, "enhanced_collection_metadata.json")
        if os.path.exists(alt):
            col_meta_path = alt
    with open(col_meta_path, "r") as f:
        col_meta = json.load(f)
    seg_map_path = os.path.join(meta_dir, "segmentation_id_map.json")
    if os.path.exists(seg_map_path):
        with open(seg_map_path, "r") as f:
            seg_map = json.load(f)
    else:
        seg_map = col_meta.get("segmentation_id_map", {})

    # Setup diffusion pipeline with optimizations
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Flash attention
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass
    dynamo.config.suppress_errors = True

    # Warm up
    dummy = Image.new('RGB', (256,256), color='white')
    with torch.no_grad():
        _ = pipe(image=dummy, prompt="test", num_inference_steps=1, image_guidance_scale=1.0)

    # Partition data
    items = list(zip(
        col_meta.get("images", []),
        col_meta.get("segmentation_masks", []),
        col_meta.get("actions", []),
        col_meta.get("rewards", []),
        col_meta.get("terminations", []),
        col_meta.get("truncations", [])
    ))
    total = len(items)
    div = args.division_id
    nd = args.num_divisions
    chunk = total // nd
    start = (div - 1) * chunk
    end = div * chunk if div < nd else total
    print(f"Processing division {div}/{nd}: items {start} to {end-1}")

    # Detect arm/cube IDs
    arm_ids, cube_ids = [], []
    for sid_str, info in seg_map.items():
        sid = int(sid_str)
        if sid == 0:
            continue
        nm = info.get("name", "").lower()
        cls_name = info.get("class_name", "").lower()
        if any(x in nm for x in ("arm","gripper")) or "link" in cls_name:
            arm_ids.append(sid)
        elif "cube" in nm or "object" in nm:
            cube_ids.append(sid)
    all_ids = arm_ids + cube_ids
    enhanced = enhance_prompt(args.prompt, bool(arm_ids), bool(cube_ids))

    res = {"original_images":[],"segmentation_masks":[],"masked_images":[],"edited_images":[],
           "actions":[],"rewards":[],"terminations":[],"truncations":[],
           "arm_ids":arm_ids,"cube_ids":cube_ids,"prompt":args.prompt,
           "enhanced_prompt":enhanced,"division_id":div,"start_index":start,"end_index":end}

    with torch.no_grad(), torch.cuda.amp.autocast():
        for img_p, seg_p, act, rew, term, trunc in tqdm(items[start:end], desc=f"Div {div}"):
            try:
                img = Image.open(os.path.join(inp, img_p))
                img = ImageOps.exif_transpose(img).convert("RGB")
                orig_np = np.array(img)
                seg_np = np.array(Image.open(os.path.join(inp, seg_p)))
                mask = create_object_mask(seg_np, all_ids)
                masked_np = orig_np.copy()
                if args.inpaint_method=="black":
                    masked_np[mask] = 0
                elif args.inpaint_method=="white":
                    masked_np[mask] = 255
                elif args.inpaint_method=="blur":
                    from scipy import ndimage
                    for c in range(3):
                        _, idx = ndimage.distance_transform_edt(~mask, return_indices=True)
                        masked_np[...,c][mask] = orig_np[...,c][idx[0][mask], idx[1][mask]]
                else:
                    masked_np[mask] = orig_np[mask]
                fname = os.path.basename(img_p)
                Image.fromarray(masked_np).save(os.path.join(out,"masked_images",fname))

                gen = torch.Generator(device="cuda").manual_seed(42)
                edited = pipe(
                    image=Image.fromarray(masked_np.astype(np.uint8)),
                    prompt=enhanced,
                    num_inference_steps=args.num_steps,
                    image_guidance_scale=args.guidance_scale,
                    generator=gen
                ).images[0]
                edt_np = np.array(edited)
                edt_np[mask] = orig_np[mask]
                Image.fromarray(edt_np).save(os.path.join(out,"images",fname))

                res["original_images"].append(img_p)
                res["segmentation_masks"].append(seg_p)
                res["actions"].append(act)
                res["rewards"].append(rew)
                res["terminations"].append(term)
                res["truncations"].append(trunc)
                res["masked_images"].append(f"masked_images/{fname}")
                res["edited_images"].append(f"images/{fname}")
            except Exception as e:
                print(f"Error on {img_p}: {e}")
    
    with open(os.path.join(out,"metadata/processed_metadata.json"),"w") as f:
        json.dump(res,f,indent=2)
    print(f"Done division {div}. Outputs in {out}")


if __name__ == "__main__":
    main()
