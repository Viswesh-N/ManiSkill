#!/usr/bin/env python3
"""
Decompose Panda robot visual meshes using CoACD.
This script will decompose the visual meshes (GLB files) for the Panda robot
to create simplified versions for curriculum learning.
"""

import os
import sys
import trimesh
import coacd
import numpy as np
from pathlib import Path

def convert_glb_to_obj(glb_path, obj_path):
    """Convert GLB file to OBJ format."""
    print(f"Converting {glb_path} to OBJ...")
    mesh = trimesh.load(glb_path, force='mesh')
    mesh.export(obj_path)
    return mesh

def decompose_mesh_with_coacd(obj_path, threshold=0.1, max_convex_hull=-1, mcts_iterations=150, mcts_max_depth=4, mcts_nodes=25):
    """
    Decompose mesh using CoACD.
    
    Parameters (ADJUST THESE FOR DIFFERENT SIMPLIFICATION LEVELS):
    
    CURRENT (Moderate - recognizable robot):
    - threshold=0.1, max_convex_hull=-1, mcts_iterations=150, mcts_max_depth=4, mcts_nodes=25
    
    MORE AGGRESSIVE (blockier, more obvious simplification):
    - threshold=0.3, max_convex_hull=5, mcts_iterations=100, mcts_max_depth=3, mcts_nodes=15
    
    LESS AGGRESSIVE (very subtle, closer to original):
    - threshold=0.05, max_convex_hull=-1, mcts_iterations=200, mcts_max_depth=5, mcts_nodes=30
    
    Key parameters:
    - threshold: concavity threshold (0.01~1). Higher = coarser/fewer parts.
    - max_convex_hull: max number of parts per link (-1 = no limit)
    - mcts_iterations: more = better quality but slower
    - mcts_max_depth: deeper search = better decomposition
    - mcts_nodes: more nodes = better exploration
    """
    print(f"Decomposing {obj_path} with threshold={threshold}, max_hulls={max_convex_hull}...")
    
    # Load mesh
    mesh = trimesh.load(obj_path, force='mesh')
    
    # Create CoACD mesh
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    
    # Run CoACD with specified parameters
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        mcts_nodes=mcts_nodes,
        preprocess_mode='auto'
    )
    
    print(f"  Decomposed into {len(parts)} convex parts")
    
    # Combine all parts into a single mesh
    if len(parts) == 0:
        print("  Warning: No parts generated, using original mesh")
        return mesh
    
    vertices_list = []
    faces_list = []
    vertex_offset = 0
    
    for part in parts:
        # Each part is [vertices, faces]
        vertices = np.array(part[0])  # vertices
        faces = np.array(part[1]) + vertex_offset  # faces
        
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
    
    return combined_mesh

def process_panda_meshes(threshold=0.1, max_convex_hull=-1, mcts_iterations=150, 
                         mcts_max_depth=4, mcts_nodes=25, backup=True):
    """
    Process all Panda visual meshes.
    
    Parameters:
    - threshold: CoACD threshold parameter (higher = more aggressive)
    - max_convex_hull: max parts per link (-1 = no limit)
    - mcts_iterations: MCTS iterations (more = better quality)
    - mcts_max_depth: MCTS depth (deeper = better decomposition)
    - mcts_nodes: MCTS nodes (more = better exploration)
    - backup: whether to backup original meshes
    """
    # Paths
    mesh_dir = Path("/home/viswesh/grid/curriculum/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual")
    temp_dir = Path("/tmp/panda_mesh_decomposition")
    temp_dir.mkdir(exist_ok=True)
    
    # Backup directory
    if backup:
        backup_dir = mesh_dir.parent / "visual_original_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backing up original meshes to {backup_dir}")
    
    # List of visual mesh files to process
    visual_meshes = [
        "link0.glb",
        "link1.glb",
        "link2.glb",
        "link3.glb",
        "link4.glb",
        "link5.glb",
        "link6.glb",
        "link7.glb",
        "hand.glb",
        "finger.glb"
    ]
    
    for mesh_name in visual_meshes:
        print(f"\n{'='*60}")
        print(f"Processing {mesh_name}")
        print('='*60)
        
        glb_path = mesh_dir / mesh_name
        if not glb_path.exists():
            print(f"Warning: {glb_path} not found, skipping...")
            continue
        
        # Backup original
        if backup:
            backup_path = backup_dir / mesh_name
            if not backup_path.exists():
                import shutil
                shutil.copy2(glb_path, backup_path)
                print(f"Backed up to {backup_path}")
        
        # Temporary paths
        obj_path = temp_dir / f"{mesh_name.replace('.glb', '.obj')}"
        decomposed_glb = temp_dir / f"{mesh_name.replace('.glb', '_decomposed.glb')}"
        
        try:
            # Convert GLB to OBJ
            convert_glb_to_obj(glb_path, obj_path)
            
            # Decompose with CoACD
            decomposed_mesh = decompose_mesh_with_coacd(
                obj_path, 
                threshold=threshold,
                max_convex_hull=max_convex_hull,
                mcts_iterations=mcts_iterations,
                mcts_max_depth=mcts_max_depth,
                mcts_nodes=mcts_nodes
            )
            
            # Save decomposed mesh as GLB
            print(f"Saving decomposed mesh as {decomposed_glb}...")
            decomposed_mesh.export(decomposed_glb)
            
            # Replace original GLB with decomposed version
            print(f"Replacing original mesh at {glb_path}...")
            import shutil
            shutil.copy2(decomposed_glb, glb_path)
            
            print(f"✓ Successfully processed {mesh_name}")
            
        except Exception as e:
            print(f"✗ Error processing {mesh_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print('='*60)
    print(f"Original meshes backed up to: {backup_dir}")
    print(f"Temporary files in: {temp_dir}")

if __name__ == "__main__":
    print("Starting Panda robot mesh decomposition with CoACD")
    print()
    
    # ============================================================================
    # CHOOSE YOUR SIMPLIFICATION LEVEL - SWAP THESE PARAMETERS:
    # ============================================================================
    
    # OPTION 1: MODERATE (current - recognizable robot, 79% fewer vertices)
    # Good for: Starting point for curriculum learning
    threshold = 0.1
    max_convex_hull = -1
    mcts_iterations = 150
    mcts_max_depth = 4
    mcts_nodes = 25
    
    # OPTION 2: AGGRESSIVE (blockier, more obviously simplified)
    # Good for: Easier visual learning, very fast rendering
    # Uncomment below to use:
    # threshold = 0.3
    # max_convex_hull = 5
    # mcts_iterations = 100
    # mcts_max_depth = 3
    # mcts_nodes = 15
    
    # OPTION 3: SUBTLE (very close to original, minimal simplification)
    # Good for: High-fidelity curriculum end stage
    # Uncomment below to use:
    # threshold = 0.05
    # max_convex_hull = -1
    # mcts_iterations = 200
    # mcts_max_depth = 5
    # mcts_nodes = 30
    
    # ============================================================================
    
    print("Using parameters:")
    print(f"  - threshold: {threshold}")
    print(f"  - max_convex_hull: {max_convex_hull}")
    print(f"  - mcts_iterations: {mcts_iterations}")
    print(f"  - mcts_max_depth: {mcts_max_depth}")
    print(f"  - mcts_nodes: {mcts_nodes}")
    print()
    
    process_panda_meshes(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        mcts_nodes=mcts_nodes,
        backup=True
    )

