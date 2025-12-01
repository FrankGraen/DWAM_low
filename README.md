# DWAM
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

## Installation
### 1. Create and activate a new python environment
We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) to manage your Python environment.
Alternatively, you can use Python's built-in [venv](https://docs.python.org/3/library/venv.html) module to create a virtual environment:
```bash
# conda
conda create -n DWAM python=3.10
conda activate DWAM
# or venv
python3.10 -m venv env_DWAM
source env_DWAM/bin/activate
```

### 2. Setup Isaac Lab
Set up Isaac Lab by following the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 3. Clone the Repo
```bash
git clone https://github.com/ai4ce/DWAM.git
cd DWAM
```

### 4. Install Dependencies  
Make sure your python environment is activated, then run:
```bash
python -m pip install -e .[all]
```
This will install all required dependencies.

## Quick start
### 1.Training a Policy
```bash
python scripts/train.py --num_envs=<number_of_environments> --enable_cameras --headless 
```
To visualize the training process with the GUI:

- Remove the `--headless` flag.
```bash
python scripts/train.py --num_envs=<number_of_environments> --enable_cameras
```
- (Optional) Enable livestreaming for remote viewing (details seen in [isaaclab.app documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html#module-isaaclab.app))：
```bash
python scripts/train.py --num_envs=<number_of_environments> --enable_cameras --livestream 2
```

### 2.Evaluating a Trained Policy
```bash
python scripts/play.py --num_envs=<number_of_environments> --enable_cameras --checkpoint="/path/to/your/policy/agent.pt"
```
Similarily, you can use `--livestream 2` mode

### 3.Debug Mode with Keyboard Control
```bash
python scripts/control.py --num_envs 1 --enable_cameras
```