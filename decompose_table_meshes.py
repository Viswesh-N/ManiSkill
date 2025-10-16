#!/usr/bin/env python3
"""
Decompose table visual meshes using CoACD.
This script will decompose the visual meshes (GLB files) for the table
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

def decompose_mesh_with_coacd(obj_path, threshold=0.1, max_convex_hull=-1, mcts_iterations=150,
                               mcts_max_depth=4, mcts_nodes=25, max_ch_vertex=256,
                               preprocess_resolution=50, decimate=False):
    """
    Decompose mesh using CoACD.
    
    Parameters (ADJUST THESE FOR DIFFERENT SIMPLIFICATION LEVELS):
    
    CURRENT (Moderate - recognizable table):
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
    - max_ch_vertex: max vertices per convex hull (lower = simpler shapes)
    - preprocess_resolution: voxel resolution for preprocessing (lower = coarser)
    - decimate: enable mesh decimation (reduces vertex count)
    """
    print(f"Decomposing {obj_path} with threshold={threshold}, max_hulls={max_convex_hull}, max_verts={max_ch_vertex}...")
    
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
        preprocess_mode='auto',
        preprocess_resolution=preprocess_resolution,
        max_ch_vertex=max_ch_vertex,
        decimate=decimate
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

def process_table_meshes(threshold=0.1, max_convex_hull=-1, mcts_iterations=150,
                         mcts_max_depth=4, mcts_nodes=25, max_ch_vertex=256,
                         preprocess_resolution=50, decimate=False, backup=True,
                         output_level="intermediate"):
    """
    Process table visual meshes.
    
    Parameters:
    - threshold: CoACD threshold parameter (higher = more aggressive)
    - max_convex_hull: max parts per link (-1 = no limit)
    - mcts_iterations: MCTS iterations (more = better quality)
    - mcts_max_depth: MCTS depth (deeper = better decomposition)
    - mcts_nodes: MCTS nodes (more = more exploration)
    - output_level: "intermediate" or "extreme" - which directory to save to
    """
    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Paths - use absolute path from script location
    mesh_dir = script_dir / "mani_skill" / "utils" / "scene_builder" / "table" / "assets"
    temp_dir = Path("/tmp/table_mesh_decomposition")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Looking for table mesh in: {mesh_dir}")
    
    # Backup directory
    if backup:
        backup_dir = mesh_dir / "original_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backing up original meshes to {backup_dir}")
    
    # Output directory based on level
    output_dir = mesh_dir / f"table_{output_level}"
    output_dir.mkdir(exist_ok=True)
    print(f"Will save simplified meshes to {output_dir}")
    
    # Process table.glb
    mesh_name = "table.glb"
    print(f"\n{'='*60}")
    print(f"Processing {mesh_name}")
    print('='*60)
    
    glb_path = mesh_dir / mesh_name
    if not glb_path.exists():
        print(f"Error: {glb_path} not found!")
        return
    
    # Backup original
    if backup:
        backup_path = backup_dir / mesh_name
        if not backup_path.exists():
            import shutil
            shutil.copy2(glb_path, backup_path)
            print(f"Backed up to {backup_path}")
    
    # Temporary paths
    obj_path = temp_dir / f"{mesh_name.replace('.glb', '.obj')}"
    decomposed_glb = output_dir / mesh_name
    
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
            mcts_nodes=mcts_nodes,
            max_ch_vertex=max_ch_vertex,
            preprocess_resolution=preprocess_resolution,
            decimate=decimate
        )
        
        # Save decomposed mesh as GLB
        print(f"Saving decomposed mesh as {decomposed_glb}...")
        decomposed_mesh.export(decomposed_glb)
        
        print(f"✓ Successfully processed {mesh_name}")
        
    except Exception as e:
        print(f"✗ Error processing {mesh_name}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print('='*60)
    print(f"Original mesh backed up to: {backup_dir}")
    print(f"Simplified mesh saved to: {output_dir}")
    print(f"Temporary files in: {temp_dir}")

if __name__ == "__main__":
    print("Starting table mesh decomposition with CoACD")
    print()
    
    # Check if level argument is provided
    if len(sys.argv) > 1 and sys.argv[1] in ['intermediate', 'extreme']:
        level = sys.argv[1]
    else:
        print("Usage: python decompose_table_meshes.py [intermediate|extreme]")
        print("Defaulting to 'intermediate'")
        level = 'intermediate'
    
    # ============================================================================
    # SIMPLIFICATION LEVELS
    # ============================================================================
    
    if level == 'intermediate':
        print("Creating INTERMEDIATE simplification level")
        threshold = 0.1
        max_convex_hull = -1
        mcts_iterations = 150
        mcts_max_depth = 4
        mcts_nodes = 25
        max_ch_vertex = 256
        preprocess_resolution = 50
        decimate = False
    else:  # extreme
        print("Creating EXTREME simplification level")
        threshold = 1.0
        max_convex_hull = 1
        mcts_iterations = 50
        mcts_max_depth = 2
        mcts_nodes = 5
        max_ch_vertex = 32
        preprocess_resolution = 20
        decimate = True
    
    # ============================================================================
    
    print(f"Level: {level}")
    print("Using parameters:")
    print(f"  - threshold: {threshold}")
    print(f"  - max_convex_hull: {max_convex_hull}")
    print(f"  - mcts_iterations: {mcts_iterations}")
    print(f"  - mcts_max_depth: {mcts_max_depth}")
    print(f"  - mcts_nodes: {mcts_nodes}")
    print(f"  - max_ch_vertex: {max_ch_vertex}")
    print(f"  - preprocess_resolution: {preprocess_resolution}")
    print(f"  - decimate: {decimate}")
    print()

    process_table_meshes(
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        mcts_nodes=mcts_nodes,
        max_ch_vertex=max_ch_vertex,
        preprocess_resolution=preprocess_resolution,
        decimate=decimate,
        backup=True,
        output_level=level
    )


