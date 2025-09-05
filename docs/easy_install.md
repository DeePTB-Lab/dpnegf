# Installation Guide

This guide will help you install DPNEGF, a Python package that integrates the Deep Learning Tight-Binding (**DeePTB**) approach with the Non-Equilibrium Green's Function (**NEGF**) method.

## Prerequisites

Before installing DPNEGF, ensure you have the following prerequisites:
  - Git
  - Python 3.9 to 3.12.
  - Torch 2.0.0 to 2.5.1 ([PyTorch Installation](https://pytorch.org/get-started/locally)).
  - [DeePTB](https://github.com/deepmodeling/DeePTB) ≥ 2.1.1

## Installation Methods

### From Source
We recommend installing **DPNEGF** within a dedicated virtual environment to manage dependencies effectively. Ensure that both DPNEGF and DeePTB are installed in the same environment for compatibility.

1. Clone the repository:
    ```bash
    git clone https://github.com/DeePTB-Lab/dpnegf.git
    ```
2. Navigate to the root directory and install DPNEGF:
    ```bash
    cd dpnegf
    pip install .
    ```


### Additional Tips

- Keep your DPNEGF installation up-to-date by pulling the latest changes from the repository and re-installing.
- If you encounter any issues during installation, consult the [DPNEGF documentation](https://deeptb-lab.github.io/dpnegf/) or seek help from the community.

## Contributing

We welcome contributions to DeePTB. If you are interested in contributing, please read our [contributing guidelines](https://deeptb-lab.github.io/dpnegf/CONTRIBUTING.html).

<!-- ## License

DPNEGF is open-source software released under the [LGPL-3.0](https://github.com/deepmodeling/DeePTB/blob/main/LICENSE) provided in the repository. -->

