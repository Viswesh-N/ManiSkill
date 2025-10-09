#!/usr/bin/env python3
"""
Restore the original Panda robot visual meshes from backup.
"""

import shutil
from pathlib import Path

def restore_original_meshes():
    """Restore original Panda visual meshes from backup."""
    
    mesh_dir = Path("/home/viswesh/grid/curriculum/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual")
    backup_dir = Path("/home/viswesh/grid/curriculum/ManiSkill/mani_skill/assets/robots/panda/franka_description/meshes/visual_original_backup")
    
    if not backup_dir.exists():
        print(f"Error: Backup directory not found at {backup_dir}")
        return False
    
    print("Restoring original Panda visual meshes...")
    print("=" * 60)
    
    # List of mesh files
    mesh_files = [
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
    
    restored_count = 0
    
    for mesh_file in mesh_files:
        backup_path = backup_dir / mesh_file
        target_path = mesh_dir / mesh_file
        
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, target_path)
                print(f"✓ Restored {mesh_file}")
                restored_count += 1
            except Exception as e:
                print(f"✗ Error restoring {mesh_file}: {e}")
        else:
            print(f"⚠ Backup not found for {mesh_file}")
    
    print("=" * 60)
    print(f"Restoration complete! Restored {restored_count}/{len(mesh_files)} files")
    
    return restored_count == len(mesh_files)

if __name__ == "__main__":
    success = restore_original_meshes()
    exit(0 if success else 1)

