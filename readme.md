# Project setup

1. Ensure you have python 3.12
2. Create a virtual environment
```
python -m venv .venv
```
3. Activate the virtual environment
```
source .venv/bin/activate
```
4. For GPU version of pytorch
```
pip install -r requirements.txt
```
For CPU fersion of pytorch first install pip-tools
```
python -m pip install pip-tools
```
Then compile and install the requirements
```
pip-compile requirements.in --extra-index-url https://download.pytorch.org/whl/cpu
pip-sync requirements.txt
```

# Logs

After activating the virtual environment, run
```
tensorboard --logdir=src/logs/tensorboard
```
You can access the logs through browser: http://localhost:6006/

# Training a model
First you need to represent your dataset as a pytorch tensor (save it in a `.pt` file). This can be done by running the notebook `dataset_creation.ipynb`


Then you can run the `invoke_training.sh` script, but make sure you specify the appropriate arguments.
For details of the arguments run
```
python -m src.main --help
```