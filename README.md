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