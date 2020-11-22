# smoke-detection-net
Smoke detection network for the [ProjectX 2020](https://www.projectx2020.com/) competition.

## Setup
Requires a linux machine with a NVIDIA GPU. Clone this repo and then run:
```
cd scripts && source ./setup_workspace.sh
```
This gives the option to download miniconda (required if not yet installed), sets up a new conda environment, installs the necessary packages, gives the option to download fake data,
and generates the metadata json files for the data loaders. NOTE, restarting the shell might be required
for the conda environment to be activated.

## Running pipeline
Modify parameters in the set_params() method inside main and then run:
```
python main.py
```

## Jupyter Notebook on a GCP instance
Follow this [blog](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)
to configure running Jupyter Notebooks on a GCP compute instance.

The following is the command to run Jupyter (replace '5000' with the configured port):
```
jupyter notebook --no-browser --port=5000
```

## TensorBoard
To run TensorBoard during training on a remote VM, `cd` to the root directory of this repo and execute (replace '5000' with configured port):
```
tensorboard --logdir output --port 5000 --bind_all
```

Assumes the Jupyter Notebook steps have been followed, i.e. external static IP, TCP port, etc.
