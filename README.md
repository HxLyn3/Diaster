# Diaster: Episodic Return Decomposition by Difference of Implicitly Assigned Sub-trajectory Reward

This is the code for the paper "Episodic Return Decomposition by Difference of Implicitly Assigned Sub-trajectory Reward".

## Installation instructions

Install Python environment with:

```bash
conda create -n diaster python=3.9 -y
conda activate diaster
conda install pytorch cudatoolkit=11.3 -c pytorch -y
pip install -r ./requirements.txt
```

## Run an experiment 

```shell
python3 main.py --env-name=[Env name] 
```

The config files act as defaults for a task. 

They are all located in `config`.
`--env-name` refers to the config files in `config/` including Hopper-v3, Walker2d-v3, Swimmer-v3, Humanoid-v3, HumanoidStandup-v2.

All results will be stored in the `result` folder.

For example, run Diaster on Hopper:

```bash
python main.py --env-name=Hopper-v3
```
