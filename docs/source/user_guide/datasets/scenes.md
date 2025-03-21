# Scene Datasets

We provide a command line tool to download scene datasets (typically adapted from the original dataset).

ManiSkill can build any scene provided assets are provided. ManiSkill out of the box provides code and download links to use [RoboCasa](https://github.com/robocasa/robocasa), [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/), and [AI2THOR](https://github.com/allenai/ai2thor) set of scenes (shown below). These are picked because of their generally high modelling quality and interactability. 

```{figure} images/scene_examples.png
```

ManiSkill is also capable of heterogeneous GPU simulation where each parallel environment can have different objects and textures. An example from a bird-eye view generated by the simulator is shown with the RoboCasa kitchen scenes below 

```{figure} images/heterogeneous_robocasa_sim.png
```

To get started with these scenes, you can download them using the following commands. Note that if you use these scenes in your work please cite ManiSkill3 in addition to the scene dataset authors.

```bash
# list all scene datasets available for download
python -m mani_skill.utils.download_asset --list "scene"
python -m mani_skill.utils.download_asset ReplicaCAD # small scene and fast to download
python -m mani_skill.utils.download_asset RoboCasa # lots of procedurally generated scenes and fast to download
python -m mani_skill.utils.download_asset AI2THOR # lots of scenes and slow to download
```

## Exploring the Scene Datasets

To explore the scene datasets, you can provide an environment ID and a seed (to change which scene is sampled if there are several available) and run the random action script. Shown below are the various environment configured already to enable you to play with RoboCasa, ReplicaCAD, and ArchitecTHOR (an AI2THOR variant).

```bash
python -m mani_skill.examples.demo_random_action \
  -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="rgb_array" --record-dir="videos" # run headless and save video

python -m mani_skill.examples.demo_random_action \
  -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="human" \
  -s 3 # open a GUI and sample a scene with seed 3

# load 4 envs and sample scenes with seeds 0, 1, 2, 3
# use fetch robot
python -m mani_skill.examples.demo_random_action \
  -e "RoboCasaKitchen-v1" \
  -n 4 -s 0 1 2 3 \
  --render-mode="human" -r "fetch"
```

You can also pass `-r "none"` to run the environments without any agents.

## Training on the Scene Datasets

Large scene datasets with hundreds of objects like ReplicaCAD and AI2THOR can be used to train more general purpose robots/agents and also serve as synthetic data generation sources. We are still in the process of providing more example code and documentation about how to best leverage these scene datasets but for now we provide code to explore and interact with the scene datasets.

### Reinforcement Learning / Imitation Learning

We are currently in the process of building task code similar to the ReplicaCAD Rearrange challenge and will open source that when it is complete. Otherwise at the moment there are not any trainable tasks with defined success/fail conditions and/or rewards that use any of the big scene datasets.

<!-- ### Computer Vision / Synthetic 2D/3D Data Generation (WIP) -->
