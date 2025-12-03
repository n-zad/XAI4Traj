# Traj-XAI: Explainable AI for Trajectory Data

A Python package for applying explainable AI (XAI) techniques to trajectory data. This package provides tools for segmenting trajectories, applying perturbations, and generating explanations for black box models.

## About This Fork
This repository is a fork of the original XAI4Traj project:
https://github.com/DAIR-Group/XAI4Traj

Modifications in this fork:
- Included temporal data in the explanation process
- Adjusted hyperparameters of the surrogate linear model
- Implemented a new perturbation method using a pretrained model from *Social GAN* (cited below)
- Improved Windows OS compatibility
- Made minor bug fixes

## Features

- **Trajectory Segmentation**: Multiple methods for dividing trajectories into meaningful segments
  - RDP (Ramer-Douglas-Peucker) segmentation
  - MDL (Minimum Description Length) segmentation
  - Sliding window approach
  - Random segmentation

- **Trajectory Perturbation**: Methods to apply controlled modifications to trajectory segments
  - Gaussian noise perturbation
  - Scaling perturbation
  - Rotation perturbation
  - GAN perturbation (experimental)

- **Model Explanation**: Generate explanations for trajectory classifications from black box models  
  - Including option to include temporal data in the explanation process (currently only works with Gaussian and Scaling perturbations)

- **Evaluation**: Tools for evaluating the quality of trajectory explanations

## Installation

```bash
# Clone the repository
git clone https://github.com/buidangphuc/XAI4Traj.git
cd traj-xai

# Install the package
pip install -e .
```

## Quick Start

```python
from pactus import Dataset
from pactus.models import LSTMModel
from traj_xai import rdp_segmentation, gaussian_perturbation, run_experiments

# Load a dataset
dataset = Dataset.uci_movement_libras()
train, test = dataset.split(0.8, random_state=0)

# Train a model
model = LSTMModel(random_state=0)
model.train(train, dataset, epochs=10, batch_size=64)

# Run XAI experiments
segment_funcs = [rdp_segmentation]
perturbation_funcs = [gaussian_perturbation]
results = run_experiments(test, segment_funcs, perturbation_funcs, model)
```

## Examples

See the `notebooks` directory for example usage:

- `basic_example.ipynb`: Demonstrates basic usage of the package
  - `basic_example_colab.ipynb`: Alternate notebook for Google Colab
- `final_experiment.ipynb`: Runs through multiple experiments and calculates their fidelity metrics
  - `final_experiment_colab.ipynb`: Alternate notebook for Google Colab
- `comparison.ipynb`: Compares different segmentation and perturbation methods
- `get_visual.ipynb`: Generate visualizations for trajectory explanations
- `test_gan_functionality.ipynb`: Combines `basic_example.ipynb` and `final_experiment.ipynb` to expirement with GAN perturbations
- `test_time_functionality.ipynb`: Copy of `final_experiment.ipynb` to expirement with time-aware explanations

## Requirements

See `requirements.txt` for detailed dependencies.

## Citations

Original repository:

```
@software{traj_xai,
  author = {Le Xuan Tung, Bui Dang Phuc},
  title = {Traj-XAI: Explainable AI for Trajectory Data},
  url = {https://github.com/buidangphuc/XAI4Traj.git},
  year = {2025},
}
```

Associated journal article:

```
@article{TUNG2025131691,
  title = {Towards explainable trajectory classification: A segment-based perturbation approach},
  journal = {Neurocomputing},
  volume = {658},
  year = {2025},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2025.131691},
  url = {https://www.sciencedirect.com/science/article/pii/S092523122502363X},
  author = {Le Xuan Tung and Bui Dang Phuc and Vo Nguyen Le Duy},
}
```

Social GAN:

```
@inproceedings{gupta2018social,
  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  number={CONF},
  year={2018}
}
```
