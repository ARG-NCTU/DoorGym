#! /bin/bash

# For the UR5 pull task
# python3 train.py --env-name doorenv_gym-v0 --save-name ur5_pull --world-path ~/DoorGym/world_generator/ur5-pull/pull_ur5/

# For the Husky_UR5 pull task
# python3 train.py --env-name doorenv_6joints-v0 --save-name husky_ur5_pull --world-path ~/DoorGym/world_generator/husky-ur5-pull/pull_husky_ur5/

# For the Husky_UR5 3dof pull task
python3 train.py --env-name doorenv-v0 --save-name husky_ur5_pull_3dof --world-path ~/DoorGym/world_generator/husky-ur5-pull/pull_husky_ur5/