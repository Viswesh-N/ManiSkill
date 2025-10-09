#!/usr/bin/env python3
"""
Compare the complexity of original vs decomposed Panda meshes.
"""

import trimesh
from pathlib import Path

def compare_meshes():
    """Compare original and decomposed mesh complexity."""
    
    visual_dir = Path("/home/viswesh/grid/curriculum/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual")
    backup_dir = Path("/home/viswesh/grid/curriculum/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual_original_backup")
    
    mesh_files = [
        "link0.glb", "link1.glb", "link2.glb", "link3.glb", "link4.glb",
        "link5.glb", "link6.glb", "link7.glb", "hand.glb", "finger.glb"
    ]
    
    print("=" * 80)
    print("PANDA ROBOT MESH COMPLEXITY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Mesh':<12} | {'Original':<25} | {'Decomposed':<25} | {'Reduction':<10}")
    print(f"{'File':<12} | {'Vertices':>8} {'Faces':>8} {'Size':>7} | {'Vertices':>8} {'Faces':>8} {'Size':>7} | {'Vertices':>5} {'Faces':>5}")
    print("-" * 80)
    
    total_orig_verts = 0
    total_orig_faces = 0
    total_decomp_verts = 0
    total_decomp_faces = 0
    
    for mesh_file in mesh_files:
        orig_path = backup_dir / mesh_file
        decomp_path = visual_dir / mesh_file
        
        if not orig_path.exists() or not decomp_path.exists():
            continue
        
        # Load meshes
        orig_mesh = trimesh.load(orig_path, force='mesh')
        decomp_mesh = trimesh.load(decomp_path, force='mesh')
        
        # Get stats
        orig_verts = len(orig_mesh.vertices)
        orig_faces = len(orig_mesh.faces)
        orig_size = orig_path.stat().st_size
        
        decomp_verts = len(decomp_mesh.vertices)
        decomp_faces = len(decomp_mesh.faces)
        decomp_size = decomp_path.stat().st_size
        
        # Calculate reductions
        vert_reduction = (1 - decomp_verts / orig_verts) * 100 if orig_verts > 0 else 0
        face_reduction = (1 - decomp_faces / orig_faces) * 100 if orig_faces > 0 else 0
        
        # Update totals
        total_orig_verts += orig_verts
        total_orig_faces += orig_faces
        total_decomp_verts += decomp_verts
        total_decomp_faces += decomp_faces
        
        # Format size
        def format_size(size):
            if size > 1024 * 1024:
                return f"{size / (1024 * 1024):.1f}M"
            else:
                return f"{size / 1024:.0f}K"
        
        print(f"{mesh_file:<12} | {orig_verts:>8} {orig_faces:>8} {format_size(orig_size):>7} | "
              f"{decomp_verts:>8} {decomp_faces:>8} {format_size(decomp_size):>7} | "
              f"{vert_reduction:>4.0f}% {face_reduction:>4.0f}%")
    
    print("-" * 80)
    
    # Calculate total reductions
    total_vert_reduction = (1 - total_decomp_verts / total_orig_verts) * 100
    total_face_reduction = (1 - total_decomp_faces / total_orig_faces) * 100
    
    print(f"{'TOTAL':<12} | {total_orig_verts:>8} {total_orig_faces:>8} {'':>7} | "
          f"{total_decomp_verts:>8} {total_decomp_faces:>8} {'':>7} | "
          f"{total_vert_reduction:>4.0f}% {total_face_reduction:>4.0f}%")
    
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  Total vertices reduced by {total_vert_reduction:.1f}%")
    print(f"  Total faces reduced by {total_face_reduction:.1f}%")
    print(f"  This makes rendering faster and may help RL agents learn basic behaviors")
    print("=" * 80)

if __name__ == "__main__":
    compare_meshes()

