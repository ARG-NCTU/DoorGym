#! /bin/bash

# For the UR5 push task
# python3 train.py --env-name doorenv_gym-v0 --save-name ur5_push --world-path ~/DoorGym/world_generator/ur5-push/pull_ur5/

# For the Husky_UR5 push task
# python3 train.py --env-name doorenv_6joints-v0 --save-name husky_ur5_push --world-path ~/DoorGym/world_generator/husky-ur5-push/pull_husky_ur5/

# For the Husky_UR5 3dof push task.
python3 train.py --env-name doorenv-v0 --save-name husky_ur5_push_3dof --world-path ~/DoorGym/world_generator/husky-ur5-push/pull_husky_ur5/