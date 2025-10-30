#!/bin/bash
# PPO RGB Training with 3-Level Mesh Curriculum
# Runs training sequentially with: extreme -> intermediate -> original meshes

cd /home/viswesh/grid/curriculum/ManiSkill

echo "========================================================"
echo "Starting 2-Level Mesh Curriculum Training"
echo "========================================================"
echo ""

# Level 1: EXTREME simplification (1.9K per mesh, 32 vertices max)
echo "========================================================"
echo "LEVEL 1/3: Training with EXTREME mesh simplification"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickCube-v1" \
  --simplify_robot_mesh=2 \
  --simplify_table \
  --num_envs=512 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="mesh_decomposition" \
  --exp_name="pickcube_robot_extreme_table_extreme" \
  --capture_video \
  --save_model \
  --eval_freq=25

# echo ""
# echo "Level 1 complete! Moving to Level 2..."
# echo ""

# Level 1: INTERMEDIATE simplification (96K-615K per mesh)
echo "========================================================"
echo "LEVEL 1/2: Training with INTERMEDIATE mesh simplification"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickCube-v1" \
  --simplify_robot_mesh=1 \
  --simplify_table \
  --num_envs=512 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="mesh_decomposition" \
  --exp_name="pickcube_robot_intermediate_table_intermediate" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Level 1 complete! Moving to Level 2..."
echo ""

# Level 2: ORIGINAL meshes (full fidelity)
echo "========================================================"
echo "LEVEL 2/2: Training with ORIGINAL meshes (full fidelity)"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickCube-v1" \
  --simplify_robot_mesh=0 \
  --simplify_table \
  --num_envs=512 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="mesh_decomposition" \
  --exp_name="pickcube_robot_original_table_original" \
  --capture_video \
  --save_model \
  --eval_freq=25



echo ""
echo "========================================================"
echo "All 2 levels complete!"
echo "========================================================"
echo "Check wandb for training progress and videos"
echo "Results available at:"
echo "  - runs/pickcube_robot_intermediate/"
echo "  - runs/pickcube_robot_original/"

