# Facial Misrecognition Systems: Simple Weight Manipulations Force DNNs to Err Only on Specific Persons
Implementation of https://arxiv.org/abs/2301.03118

## Setup
1. Run `scripts/install_dependencies.sh`
2. Create a `data` directory in the repo's root directory, and download all relevant datasets there
3. Run `scripts/mtcnn.py` on each dataset directory to preprocess the image files


## Usage
See `notebooks` directory for all experiments. Note that before running the MNIST visualization you must run `notebooks/pretrain/mnist_3_features.ipynb` to train a 3-feature classifier for MNIST.
