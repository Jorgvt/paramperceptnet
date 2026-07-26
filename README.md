# paramperceptnet
Parametric enhancement of PerceptNet.

## Installation

We currently provide two options for installing the package and using the model:

1. Minimal installation: Installs only the required libraries to instantiate, use and train a parametric model.

`pip install "paramperceptnet @ git+https://github.com/Jorgvt/paramperceptnet.git"`

2. Full installation: Installs all the libraries required to run the examples in `./Examples/`. This includes loading pre-trained weights and a sample dataset from HuggingFace and plotting the results.

`pip install "paramperceptnet[examples] @ git+https://github.com/Jorgvt/paramperceptnet"`


## Pre-trained models

We have uploaded a couple of pre-trained models to HuggingFace:

1. Parametric Fully Trained: (https://huggingface.co/Jorgvt/ppnet-fully-trained)
2. Parametric Bio-Fitted: (https://huggingface.co/Jorgvt/ppnet-bio-fitted)

You can easily load any of these pretrained models with a few lines of code:

```python
from paramperceptnet.pretrained import load_param_pretrained

# Load the model and its associated parameters/state
model, variables = load_param_pretrained("ppnet-bio-fitted")
state = variables["state"]
params = variables["params"]
```

For baseline models (e.g., `Jorgvt/ppnet-baseline`), use the baseline loader:

```python
from paramperceptnet.pretrained import load_baseline_pretrained

model, variables = load_baseline_pretrained("ppnet-baseline")
params = variables["params"]
```

More details on how to load them can be found in their Model Cards and in the examples provided in `./Examples/`.


## Examples

There are some notebook usage examples in the `./Examples/` folder.
