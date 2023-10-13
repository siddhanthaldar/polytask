# PolyTask: Learning Unified Policies through Behavior Distillation

[[arXiv]]() [[Project page]](https://poly-task.github.io/)

This is a repository containing the code for the paper [PolyTask: Learning Unified Policies through Behavior Distillation]().

![main_figure](https://github.com/siddhanthaldar/polytask/assets/25313941/debaa96a-94aa-44a9-a615-b5fb42023038)

## Download expert demonstrations, weights, replay buffers and environment libraries [[link]](https://drive.google.com/drive/folders/1_hvX7y4pIASdPzmKu9mODh5a__wVY1t8?usp=sharing)
The link contains the following:
- The expert demonstrations for all tasks in the paper.
- The weight files for behavior cloning (BC) and demonstration-guided RL ([ROT](https://rot-robot.github.io/)).
- The relabeled replay buffers for all tasks in the paper.
- The supporting libraries for environments (Meta-World, Franka Kitchen) in the paper.
- Extract the files provided in the link
  - set the `path/to/dir` portion of the `root_dir` path variable in `cfgs/config*.yaml` to the path of the `PolyTask` repository.
  - place the `expert_demos`, `weights` and `buffers` folders in a common directory `${data_dir}`.
  - set the `path/to/dir` portion of the `data_dir` path variable in `cfgs/config*.yaml` to the path of the common data directory.

## Instructions to set up simulation environment
- Install [Mujoco](http://www.mujoco.org/) based on the instructions given [here](https://github.com/facebookresearch/drqv2).
- Install the following libraries:
```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
- Set up Environment
  ```
  conda env create -f conda_env.yml
  conda activate polytask
  ```
- Download the Meta-World benchmark suite and its demonstrations from [here](https://drive.google.com/drive/folders/1_hvX7y4pIASdPzmKu9mODh5a__wVY1t8?usp=sharing). Install the simulation environment using the following command - 
  ```
  pip install -e /path/to/dir/metaworld
  ```
- Download the D4RL benchmark for using the Franka Kitchen environments from [here](https://drive.google.com/drive/folders/1_hvX7y4pIASdPzmKu9mODh5a__wVY1t8?usp=sharing). Install the simulation environment using the following command - 
  ```
  pip install -e /path/to/dir/d4rl
  ```


## Instructions to train models

For running the code, enter the code directory through `cd polytask` and execute the following commands.

- Train BC agent
```
python train.py agent=bc suite=dmc obs_type=features suite.task_id=[1] num_demos_per_task=10
```
```
python train.py agent=bc suite=metaworld obs_type=pixels suite/metaworld_task=hammer num_demos_per_task=1
```
```
python train.py agent=bc suite=kitchen obs_type=pixels suite.task=['task1'] num_demos_per_task=100
```
```
python train_robot.py agent=bc suite=robotgym obs_type=pixels suite/robotgym_task=boxopen num_demos_per_task=1
```
  
- Train Demo-Guided RL (ROT)
```
python train.py agent=drqv2 suite=dmc obs_type=features suite.task_id=[1] num_demos_per_task=10 load_bc=true bc_regularize=true
```
```
python train.py agent=drqv2 suite=metaworld obs_type=pixels suite/metaworld_task=hammer num_demos_per_task=1 load_bc=true bc_regularize=true
```
```
python train.py agent=drqv2 suite=kitchen obs_type=pixels suite.task=['task1'] num_demos_per_task=100 load_bc=true bc_regularize=true
```
```
python train_robot.py agent=drqv2 suite=robotgym obs_type=pixels suite/robotgym_task=boxopen num_demos_per_task=1 load_bc=true bc_regularize=true
```

- Train PolyTask
```
python train_distill.py agent=distill suite=dmc obs_type=features num_envs_to_distil=10 
```
```
python train_distill.py agent=distill suite=metaworld obs_type=pixels num_envs_to_distil=16 
```
```
python train_distill.py agent=distill suite=kitchen obs_type=pixels num_envs_to_distil=6
```
```
python train_robot_distill.py agent=distill suite=robotgym obs_type=pixels num_envs_to_distil=6
```

- Monitor results
```
tensorboard --logdir exp_local
```

## Code for baselines
The code for baseline algorithms is available in the `baselines` branch of this repository. The instructions to run the code are available in the `README.md` file of the `baselines` branch.

## Bibtex
```
@article{haldar2023polytask,
         title={PolyTask: Learning Unified Policies through Behavior Distillation},
         author={Haldar, Siddhant and Pinto, Lerrel},
         journal={arXiv preprint arXiv:2310.08573},
         year={2023}
        } 
```
