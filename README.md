# smoke-detection-net
Smoke detection network for the [ProjectX 2020](https://www.projectx2020.com/) competition.

## Setup
Requires a linux machine with a NVIDIA GPU. Clone this repo and then run `cd scripts && source ./setup_workspace.sh`.
This gives the option to download miniconda, sets up a new conda environment, installs the necessary packages, gives the option to download fake data,
and generates the metadata json files for the data loaders. NOTE, restarting the shell might be required
for the conda environment to be activated.

## Running pipeline
Modify parameters in the set_params() method inside main and then run `python main.py`
