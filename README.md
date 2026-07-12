# DPNEGF

> ⚠️ **IMPORTANT NOTICE**
> 
> This repository is no longer actively maintained. `dpnegf` has officially joined the **Deepmodeling** open-source community! All future development, updates, and maintenance (including GPU support and performance optimizations) will take place in our new repository.
> 👉 **Please visit our new home: [deepmodeling/dpnegf](https://github.com/deepmodeling/dpnegf)**
> 

**DPNEGF** is a Python package that integrates the Deep Learning Tight-Binding (**DeePTB**) approach with the Non-Equilibrium Green’s Function (**NEGF**) method, establishing an efficient quantum transport simulation framework **DeePTB-NEGF** with first-principles accuracy. 

By using DeePTB-SK or DeePTB-E3—both available within the DeePTB package—DeePTB-NEGF can compute quantum transport properties in open-boundary systems with either environment-corrected **Slater-Koster TB Hamiltonian** or **linear combination of atomic orbitals (LCAO) Kohn-Sham Hamiltonian**.


For more details, see our papers:
  1. [DPNEGF: npj Comput Mater 11, 375 (2025)](https://www.nature.com/articles/s41524-025-01853-6)
  2. [DeePTB-SK: Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
  3. [DeePTB-E3: ICLR 2025 Spotlight](https://openreview.net/forum?id=kpq3IIjUD3)


## Installation

Installing **DPNEGF** is straightforward. We recommend using a virtual environment for dependency management.

- **Requirements**
  - Git
  - DeePTB(https://github.com/deepmodeling/DeePTB) ≥ 2.1.1

- **From Source**
    1. Clone the repository:
        ```bash
        git clone https://github.com/DeePTB-Lab/dpnegf.git
        ```
    2. Navigate to the root directory and install DPNEGF:
        ```bash
        cd dpnegf
        pip install .
        ```
## Test code 

To ensure the code is correctly installed, please run the unit tests first:
```bash
pytest ./dpnegf/tests/
```
Be careful if not all tests pass!


## How to cite

The following references are required to be cited when using DPNEGF. Specifically:

- **For DPNEGF:**
  
    J. Zou, Z. Zhouyin, D. Lin, L. Zhang, S. Hou and Q. Gu, Deep Learning Accelerated Quantum Transport Simulations in Nanoelectronics: From Break Junctions to Field-Effect Transistors, npj Comput Mater 11, 375 (2025).


- **For DeePTB-SK:**

    Q. Gu, Z. Zhouyin, S. K. Pandey, P. Zhang, L. Zhang, and W. E, Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with Ab Initio Accuracy, Nat Commun 15, 6772 (2024).
  
- **For DeePTB-E3:**
  
    Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, Learning Local Equivariant Representations for Quantum Operators, In The 13th International Conference on Learning Representations (ICLR) 2025. 