#!/usr/bin/env python3
"""
Create simplified table visual mesh as a simple box.
This avoids convex decomposition artifacts and creates a clean box mesh.
"""

import os
import sys
import trimesh
import numpy as np
from pathlib import Path

def create_simple_box_mesh(original_glb_path):
    """
    Create a simple box mesh that approximates the table bounds.
    This is much simpler than convex decomposition and avoids artifacts.
    """
    print(f"Loading original mesh from {original_glb_path}...")
    original_mesh = trimesh.load(original_glb_path, force='mesh')
    
    # Get bounding box of original mesh
    bounds = original_mesh.bounds
    print(f"Original mesh bounds: {bounds}")
    
    # Create a simple box mesh matching the bounds
    # bounds[0] is min corner, bounds[1] is max corner
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    
    print(f"Creating box with extents: {extents}, center: {center}")
    
    # Create box mesh
    box = trimesh.creation.box(extents=extents)
    
    # Translate to match original center
    box.apply_translation(center)
    
    return box

def process_table_mesh(backup=True):
    """
    Create a simplified table mesh as a simple box.
    """
    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Paths - use absolute path from script location
    mesh_dir = script_dir / "mani_skill" / "utils" / "scene_builder" / "table" / "assets"
    
    print(f"Looking for table mesh in: {mesh_dir}")
    
    # Backup directory
    if backup:
        backup_dir = mesh_dir / "original_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"Backing up original meshes to {backup_dir}")
    
    # Output directory - just one simplified version
    output_dir = mesh_dir / "table_simplified"
    output_dir.mkdir(exist_ok=True)
    print(f"Will save simplified mesh to {output_dir}")
    
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
    
    # Output path
    simplified_glb = output_dir / mesh_name
    
    try:
        # Create simple box mesh
        box_mesh = create_simple_box_mesh(glb_path)
        
        # Save simplified mesh as GLB
        print(f"Saving simplified box mesh as {simplified_glb}...")
        box_mesh.export(simplified_glb)
        
        print(f"✓ Successfully created simplified table mesh")
        
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

if __name__ == "__main__":
    print("Creating simplified table mesh as a simple box")
    print()
    
    process_table_mesh(backup=True)


