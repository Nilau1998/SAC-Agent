## How to create a virtual env to run the Python code
- git clone via SSH or HTTPS
- make sure virtualenv is installed  `pip/pip3 install virtualenv`
- create a new venv `virtualenv venv`
- Activate the venv `source venv/bin/activate`
- Install the requirements.txt `pip install -r requirements.txt`
- Usually pytorch doesn't work, install that manually with `pip/pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

If you however use conda, like the IAT server does *sigh*, then run the following commands.
- conda create -n venv python=3.10
- conda activate venv
- pip3 install -r requirements.txt
- Usually pytorch doesn't work, install that manually with `pip/pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

## SAC implementation for dynamic systems
This repository is the implementation of a Soft-Actor-Critic Agent that is used to control a dynamic system via deep learning.

## Running the Python code
The Python code can be run in several ways. You can start the training, training with rendering or just render a previously trained experiment.\\
Use the following commandos:
- python3 main.py -t, this trains a new model with default config and saves all it's data in the dedicated experiment directory
- python3 main.py -tr, this trains and renders a new experiment/model with default config
- python3 main.py -r [path], this renders a previously trained experiment. Note, the experiment path has to look like `experiments/foo/experiment_12-19...` if you are already in the directory where main.py is located.
- python3 main.py -p None [n_models], this trains n models with random hp defined in the hp tuner. Set it to -pr to also render the experiments. Or just checkout the overview.csv in the corresponding experiment directory do see what the best model is and render just that as described in point 3.