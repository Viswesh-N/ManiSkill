#!/bin/bash
# PPO RGB Training with YCB Objects: 6-Case Ablation Study
# 3 Robot Levels (extreme, intermediate, original) x 2 Object Levels (simplified, original)
# Uses simplified table mesh throughout

set -e  # Exit on error

cd /home/viswesh/grid/curriculum/ManiSkill

echo "========================================================"
echo "YCB Ablation Study Setup"
echo "========================================================"
echo ""

# Step 1: Download YCB dataset if not present
echo "Step 1: Checking YCB dataset..."
YCB_DIR="$HOME/.maniskill/data/assets/mani_skill2_ycb"
if [ ! -d "$YCB_DIR" ]; then
    echo "  YCB dataset not found. Downloading..."
    python -m mani_skill.utils.download_asset ycb
    echo "  YCB dataset downloaded successfully!"
else
    echo "  YCB dataset already present at $YCB_DIR"
fi
echo ""

# Step 2: Generate simplified YCB visual meshes if not already generated
echo "Step 2: Checking simplified YCB meshes..."
# Check if at least one simplified mesh exists
SAMPLE_SIMPLIFIED="$YCB_DIR/models/002_master_chef_can/textured_simplified.obj"
if [ ! -f "$SAMPLE_SIMPLIFIED" ]; then
    echo "  Simplified meshes not found. Generating with CoACD..."
    echo "  This may take 10-30 minutes depending on your hardware..."
    python decompose_ycb_meshes.py --auto-confirm
    echo "  Simplified meshes generated successfully!"
else
    echo "  Simplified meshes already present"
fi
echo ""

echo "========================================================"
echo "Starting 6-Case YCB Ablation Study"
echo "3 Robot Levels x 2 Object Levels = 6 Cases"
echo "========================================================"
echo ""

# Case 1: Robot EXTREME + YCB SIMPLIFIED
echo "========================================================"
echo "CASE 1/6: Robot EXTREME + YCB SIMPLIFIED"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=2 \
  --simplify_table \
  --simplify_ycb_visual \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_extreme_obj_simplified" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Case 1 complete! Moving to Case 2..."
echo ""

# Case 2: Robot EXTREME + YCB ORIGINAL
echo "========================================================"
echo "CASE 2/6: Robot EXTREME + YCB ORIGINAL"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=2 \
  --simplify_table \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_extreme_obj_original" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Case 2 complete! Moving to Case 3..."
echo ""

# Case 3: Robot INTERMEDIATE + YCB SIMPLIFIED
echo "========================================================"
echo "CASE 3/6: Robot INTERMEDIATE + YCB SIMPLIFIED"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=1 \
  --simplify_table \
  --simplify_ycb_visual \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_intermediate_obj_simplified" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Case 3 complete! Moving to Case 4..."
echo ""

# Case 4: Robot INTERMEDIATE + YCB ORIGINAL
echo "========================================================"
echo "CASE 4/6: Robot INTERMEDIATE + YCB ORIGINAL"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=1 \
  --simplify_table \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_intermediate_obj_original" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Case 4 complete! Moving to Case 5..."
echo ""

# Case 5: Robot ORIGINAL + YCB SIMPLIFIED
echo "========================================================"
echo "CASE 5/6: Robot ORIGINAL + YCB SIMPLIFIED"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=0 \
  --simplify_table \
  --simplify_ycb_visual \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_original_obj_simplified" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "Case 5 complete! Moving to Case 6..."
echo ""

# Case 6: Robot ORIGINAL + YCB ORIGINAL (baseline)
echo "========================================================"
echo "CASE 6/6: Robot ORIGINAL + YCB ORIGINAL (BASELINE)"
echo "========================================================"
python examples/baselines/ppo/ppo_rgb.py \
  --env_id="PickSingleYCB-v1" \
  --simplify_robot_mesh=0 \
  --simplify_table \
  --num_envs=128 \
  --update_epochs=4 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --track \
  --wandb_project_name="PhysVizCurriculum" \
  --wandb_group="ycb_ablation" \
  --exp_name="pickycb_robot_original_obj_original" \
  --capture_video \
  --save_model \
  --eval_freq=25

echo ""
echo "========================================================"
echo "All 6 cases complete!"
echo "========================================================"
echo "Check wandb for training progress and videos"
echo "Ablation study results available in wandb group: ycb_ablation"
echo ""
echo "Cases run:"
echo "  1. Robot EXTREME + YCB SIMPLIFIED"
echo "  2. Robot EXTREME + YCB ORIGINAL"
echo "  3. Robot INTERMEDIATE + YCB SIMPLIFIED"
echo "  4. Robot INTERMEDIATE + YCB ORIGINAL"
echo "  5. Robot ORIGINAL + YCB SIMPLIFIED"
echo "  6. Robot ORIGINAL + YCB ORIGINAL (baseline)"
echo ""
