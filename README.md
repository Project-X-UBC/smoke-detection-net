# smoke-detection-net
Smoke detection network for the [ProjectX 2020](https://www.projectx2020.com/) competition.

## Setup
Requires a linux machine. Clone this repo and then run `./setup_workspace.sh`. 
This sets up the conda environment, installs the necessary packages, downloads fake data,
and generates the metadata json files for the data loaders.

## Running pipeline
Modify parameters in the set_params() method inside main and then run `python main.py`
