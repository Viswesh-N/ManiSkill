#!/usr/bin/env python3
"""
Decompose YCB object visual meshes using CoACD.
This script will create extremely simplified versions of YCB object meshes
for visual curriculum learning.
"""

import os
import sys
import trimesh
import coacd
import numpy as np
from pathlib import Path
from mani_skill import ASSET_DIR
from mani_skill.utils.io_utils import load_json

def decompose_mesh_with_coacd(obj_path, threshold=1.0, max_convex_hull=1,
                               mcts_iterations=50, mcts_max_depth=2, mcts_nodes=5,
                               max_ch_vertex=32, preprocess_resolution=20, decimate=True):
    """
    Decompose mesh using CoACD with EXTREME simplification settings.

    These are the same ultra-aggressive settings used for the Panda robot's
    extreme simplification level.
    """
    print(f"  Decomposing with CoACD (extreme simplification)...")

    # Load mesh
    mesh = trimesh.load(obj_path, force='mesh')

    # Get original vertex count
    orig_vertex_count = len(mesh.vertices)
    print(f"    Original vertices: {orig_vertex_count}")

    # Create CoACD mesh
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    # Run CoACD with extreme simplification parameters
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        mcts_nodes=mcts_nodes,
        preprocess_mode='auto',
        preprocess_resolution=preprocess_resolution,
        max_ch_vertex=max_ch_vertex,
        decimate=decimate
    )

    print(f"    Decomposed into {len(parts)} convex part(s)")

    # Combine all parts into a single mesh
    if len(parts) == 0:
        print("    Warning: No parts generated, using original mesh")
        return mesh

    vertices_list = []
    faces_list = []
    vertex_offset = 0

    for part in parts:
        vertices = np.array(part[0])
        faces = np.array(part[1]) + vertex_offset

        vertices_list.append(vertices)
        faces_list.append(faces)
        vertex_offset += len(vertices)

    # Combine into single mesh
    combined_vertices = np.vstack(vertices_list)
    combined_faces = np.vstack(faces_list)

    combined_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces
    )

    new_vertex_count = len(combined_mesh.vertices)
    reduction = (1 - new_vertex_count / orig_vertex_count) * 100
    print(f"    New vertices: {new_vertex_count} ({reduction:.1f}% reduction)")

    return combined_mesh

def process_ycb_meshes(backup=True):
    """
    Process all YCB visual meshes to create simplified versions.
    """
    # Get YCB models directory
    ycb_models_dir = Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "models"

    if not ycb_models_dir.exists():
        print(f"Error: YCB models directory not found at {ycb_models_dir}")
        print("Please download the YCB dataset first using:")
        print("  python -m mani_skill.utils.download_asset ycb")
        return

    # Get list of all YCB model IDs from the info file
    info_path = Path(ASSET_DIR) / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    model_db = load_json(info_path)
    model_ids = list(model_db.keys())

    print(f"Found {len(model_ids)} YCB models to process")
    print(f"Processing models in: {ycb_models_dir}")
    print()

    successful = 0
    failed = 0
    skipped = 0

    for model_id in model_ids:
        print(f"{'='*60}")
        print(f"Processing {model_id}")
        print('='*60)

        model_dir = ycb_models_dir / model_id
        if not model_dir.exists():
            print(f"  Warning: Directory not found, skipping...")
            skipped += 1
            continue

        # Input: textured.obj (original visual mesh)
        obj_path = model_dir / "textured.obj"
        if not obj_path.exists():
            print(f"  Warning: textured.obj not found, skipping...")
            skipped += 1
            continue

        # Output: textured_simplified.obj
        simplified_obj_path = model_dir / "textured_simplified.obj"

        # Backup directory
        if backup:
            backup_dir = model_dir / "original_backup"
            backup_dir.mkdir(exist_ok=True)

            # Backup original textured.obj if not already backed up
            backup_path = backup_dir / "textured.obj"
            if not backup_path.exists():
                import shutil
                shutil.copy2(obj_path, backup_path)
                print(f"  Backed up original to {backup_path}")

        try:
            # Decompose with CoACD
            decomposed_mesh = decompose_mesh_with_coacd(obj_path)

            # Save simplified mesh
            print(f"  Saving simplified mesh to {simplified_obj_path}...")
            decomposed_mesh.export(simplified_obj_path)

            # Add material references to the simplified OBJ file to use same textures
            # Read the saved file and prepend material references
            with open(simplified_obj_path, 'r') as f:
                content = f.read()

            # Write back with material references at the top (after the comment line)
            with open(simplified_obj_path, 'w') as f:
                lines = content.split('\n')
                f.write(lines[0] + '\n')  # Keep the trimesh comment
                f.write('mtllib material_0.mtl\n')
                f.write('usemtl material_0\n')
                f.write('\n'.join(lines[1:]))

            print(f"  Added material references for textures")

            print(f"  ✓ Successfully processed {model_id}")
            successful += 1

        except Exception as e:
            print(f"  ✗ Error processing {model_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

        print()

    print(f"{'='*60}")
    print("Processing complete!")
    print('='*60)
    print(f"Successfully processed: {successful}/{len(model_ids)}")
    print(f"Failed: {failed}/{len(model_ids)}")
    print(f"Skipped: {skipped}/{len(model_ids)}")
    print()
    print("Simplified meshes saved as 'textured_simplified.obj' in each model directory")
    if backup:
        print("Original meshes backed up to 'original_backup/' in each model directory")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decompose YCB object visual meshes using CoACD")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Skip confirmation prompt (for automated scripts)")
    args = parser.parse_args()

    print("Starting YCB object mesh decomposition with CoACD")
    print("Using EXTREME simplification settings (same as Panda robot level 2)")
    print()
    print("Parameters:")
    print("  - threshold: 1.0 (maximum)")
    print("  - max_convex_hull: 1 (single hull)")
    print("  - mcts_iterations: 50")
    print("  - mcts_max_depth: 2")
    print("  - mcts_nodes: 5")
    print("  - max_ch_vertex: 32 (ultra low)")
    print("  - preprocess_resolution: 20 (ultra low)")
    print("  - decimate: True")
    print()

    if not args.auto_confirm:
        response = input("This will process all YCB models. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    else:
        print("Auto-confirm enabled, proceeding with processing...")
        print()

    process_ycb_meshes(backup=True)
