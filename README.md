# paramperceptnet
Parametric enhancement of PerceptNet.

## Installation

We currently provide two options for installing the package and use the model:

1. Minimal installation: Installs only the required libraries to instantiate, use and train a parametric model.

`pip install "paramperceptnet git+https://github.com/Jorgvt/paramperceptnet.git"`

2. Full installation: Installs all the libraries required to run the examples in './Examples/'. This includes loading pre-trained weights and a sample dataset from HuggingFace and plotting the results.

`pip install "paramperceptnet[examples] @ git+https://github.com/Jorgvt/paramperceptnet"`


## Pre-trained models

We have uploaded a couple of pre-trained models to HuggingFace:

1. Parametric Fully Trained: (https://huggingface.co/Jorgvt/ppnet-fully-trained)
2. Parametric Bio-Fitted: (https://huggingface.co/Jorgvt/ppnet-bio-fitted)

Instructions on how to load them can be found in their Model Card and in the examples provided in `./Examples/`.

## Examples

There are some notebook usage examples in the `./Examples/` folder.
