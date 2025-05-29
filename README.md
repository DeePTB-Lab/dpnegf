# DPNEGF

**DPNEGF** is a Python package that integrates the Deep Learning Tight-Binding (**DeePTB**) approach with the Non-Equilibrium Green’s Function (**NEGF**) method, enabling efficient quantum transport simulations with first-principles accuracy. 

By using DeePTB-SK or DeePTB-E3—both available within the DeePTB package—DeePTB-NEGF can compute quantum transport properties in open-boundary systems with either environment-corrected **Slater-Koster tight-binding Hamiltonian** or **linear combination of atomic orbitals (LCAO) Kohn-Sham Hamiltonian**.


For more details, see our papers:
  1. [DeePTB-NEGF: arXiv:2411.08800v2](https://arxiv.org/abs/2411.08800v2)
  2. [DeePTB-SK: Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
  3. [DeePTB-E3: arXiv:2407.06053](https://arxiv.org/pdf/2407.06053)


## Installation

Installing **DPNEGF** is straightforward. We recommend using a virtual environment for dependency management.

- **Requirements**
  - Git
  - DeePTB(https://github.com/deepmodeling/DeePTB) 

- **From Source**
    1. Clone the repository:
        ```bash
        git clone https://github.com/DeePTB-Lab/dpnegf.git
        ```
    2. Navigate to the root directory and install DeePTB-NEGF:
        ```bash
        cd DeePTB-negf
        pip install .
        ```

## How to cite

The following references are required to be cited when using DeePTB-NEGF. Specifically:

- **For DeePTB-NEGF:**
  
    J. Zou, Z. Zhouyin, D. Lin, L. Zhang, S. Hou and Q. Gu, Deep Learning Accelerated Quantum Transport Simulations in Nanoelectronics: From Break Junctions to Field-Effect Transistors. arXiv:2411.08800 (2024).


- **For DeePTB-SK:**

    Q. Gu, Z. Zhouyin, S. K. Pandey, P. Zhang, L. Zhang, and W. E, Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with Ab Initio Accuracy, Nat Commun 15, 6772 (2024).
  
- **For DeePTB-E3:**
  
    Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, Learning Local Equivariant Representations for Quantum Operators, In The 13th International Conference on Learning Representations (ICLR) 2025. 