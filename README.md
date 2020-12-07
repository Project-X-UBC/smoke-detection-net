# smoke-detection-net
This project is part of our submission for [ProjectX 2020](https://www.projectx2020.com/),
a 3-month machine learning competition with a focus on climate change.

### Summary of project

In this work we present a multi-label image classifier for a smoke-detection task. 
Our approach is to divide the image into a KxK grid and predict the cells of the grid containing smoke.
We achieve 3 − 4% improvement over baseline for a 4x4 mesh and empirically establish upper bounds on grid size resolution.
As a secondary contribution, we release the first smoke-annotated video dataset which consists of 139 hours of footage 
from PTZ cameras across 678 videos. 

![alt-text-1](figures/im1_raw.png "raw-image") ![alt-text-2](figures/im1_gr_2.png "mesh-size-2x2") ![alt-text-3](figures/im1_gr_4.png "mesh-size-4x4") ![alt-text-4](figures/im1_gr_8.png "mesh-size-8x8")

Above are sample predictions with mesh sizes 2x2, 4x4, and 8x8 (TP = green, FP = blue, TN = unshaded).

The annotated videos scraped from the [Nevada Seismological Laboratory](https://www.youtube.com/user/nvseismolab/featured) 
YouTube channel are available for download [here](https://archive.org/details/smoke_ubc_projectx). 
Our pipeline is built on top of [Detectron2](https://github.com/facebookresearch/detectron2). Below are various instructions
for setting up the workspace and running the pipeline.

## Installation
Requires a linux machine with a NVIDIA GPU. Clone this repo and then run:
```
cd scripts && source ./setup_workspace.sh
```
This gives the option to download miniconda (required if not yet installed), sets up a new conda environment, installs the necessary packages.
and generates sample metadata json files for the data loaders. NOTE, restarting the shell might be required
for the conda environment to be activated.

## Running pipeline
Modify parameters in the `set_params()` method inside **main.py** and then run:
```
python main.py
```

Data is generally pulled from the **data/** directory and model outputs are stored inside **output/**.
Make sure to specify a new directory name for each training run by setting the parameter *output_dir* inside `main.py`.

## Generating dataset dicts
Detectron2 expects a specific format for the metadata json files. The script `make_real_data_json.py` 
takes care of this step. It takes in as an argument the directory containing image data and produces a json file for the 
train, validation, and test set. Note, we use `GroupShuffleSplit` to ensure frames from the same video are within the same 
split.
 
## Notebooks
The notebook [visualize_prediction_results.ipynb](https://github.com/Project-X-UBC/smoke-detection-net/blob/main/notebooks/visualize_prediction_results.ipynb)
is a great tool for visualizing model predictions and comparing them to ground truth labels. 

## Google Cloud Platform
We trained all of our models using the GCP compute engine and found some useful resources which we have added below.
#### Jupyter Notebook on a GCP instance
Follow this [blog](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)
to configure running Jupyter Notebooks on a GCP compute instance. The following is the command to run Jupyter (replace '5000' with the configured port):
```
jupyter notebook --no-browser --port=5000
```

#### TensorBoard
To run TensorBoard during training, `cd` to the root directory of this repo and execute (replace '5000' with configured port):
```
tensorboard --logdir output --port 5000 --bind_all
```

Assumes the Jupyter Notebook steps have been followed, i.e. external static IP, TCP port, etc.
