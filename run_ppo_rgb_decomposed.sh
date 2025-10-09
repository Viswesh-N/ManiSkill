#!/bin/bash
# PPO RGB Training with Decomposed Panda Meshes
# This will automatically use the decomposed visual meshes we just created!

cd /home/viswesh/grid/curriculum/ManiSkill

python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickCube-v1" \
  --num_envs=256 \
  --update_epochs=8 \
  --num_minibatches=8 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="mesh_decomposition" \
  --exp_name="pickcube_robot_decomposed" \
  --capture_video \
  --save_model \
  --eval_freq=10

# Explanation of key flags:
# --track: Enables wandb logging
# --wandb_project_name: PhysVizCurriculum (top-level project for physics/visual curriculum)
# --wandb_group: mesh_decomposition (this group for mesh-based curriculum)
#                Later you can add "viz_curriculum" group for other visual curriculum experiments
# --exp_name: pickcube_robot_decomposed (this specific run with decomposed Panda meshes)
# --capture_video: Save videos during evaluation
# --save_model: Save model checkpoints
# --eval_freq: Evaluate every 10 iterations

echo ""
echo "Training will use DECOMPOSED Panda meshes (79% fewer vertices!)"
echo "Check wandb for training progress and videos"

