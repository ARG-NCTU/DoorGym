# DoorGym

## Clone repo and Docker

I released docker images that can modify other packages on this and compile by yourself.

### Step1. Clone repo

```
git clone --recursive git@github.com:ARG-NCTU/DoorGym.git
```

### Step2. Docker run

Run this script to pull the docker image to your workstation.

```
source docker_run.sh
```

### Step3. Docker join

If you want to enter the same docker image, type below command.

```
source docker_join.sh
```

### Step4. Catkin_make

Execute the compile script at first time, then the other can ignore this step. 

```
source catkin_make.sh
```

### Step5. Setup environment

Make sure run this command when the terminal enter docker.

```
source environment.sh
```

## Download model weight

Run below command to download all model weight into model folder.

```
cd model
python3 download_weight.py
```

## Download dataset and knob model

This dataset is required for training, you can follow below command to download these.

```
cd world_generator
python3 download_data.py
```

If you want to make your own dataset, you can follow below steps.

### Step1. Lever and Round knobs

#### [Lever knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/leverknobs.tar.gz) (0.77 GB)
#### [Round knobs](https://github.com/PSVL/DoorGym/releases/download/v1.0/roundknobs.tar.gz) (1.24 GB)

### Step2. Setup knob and robot model

* Extract the downloaded door knob dataset under the `world_generator/door` folder.
* Place your favorite robots under the `world_generator/robot` folder. (Blue robots are there as default)

### Step3. Generate door world (e.g. pull knob and husky ur5 combination.)

```
cd ~/DoorGym/world_generator
python3 world_generator.py --knob-type pull --robot-type husky-ur5
```

Check the model by running the mujoco simulator

```
source environment.sh
cd ~/.mujoco/mujoco210/bin
./simulate [environment path]
```

More detailed instruction [here](./world_generator)

## Training

Before ready for the above setting, then you can train a model. For the training was need parameter setting, I arrange it and write a script that is easy to use.

### Training

* Pull task

  ```
    source environment.sh
    source train_pull.sh
  ```

* Push task

  ```
    source environment.sh
    source train_push.sh
  ```

The trained weights will store trained_models folder and log file will store logs folder, you can use tensorboard to see train curve.

If you want to train baseline (DoorGym) or 6 joints environment, you need to edit training script.

Edit train_pull.sh or train_push.sh parameter.

* DoorGym baseline

  ```
  --env-name doorenv_gym-v0
  ```

* 6 joints

  ```
  --env-name doorenv_6joints-v0
  ```

* 3 DOF

  ```
  --env-name doorenv-V0
  ```

More detailed parameters [here](https://github.com/ARG-NCTU/curl_navi/blob/master/04_DoorGym.ipynb)
