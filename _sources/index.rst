.. DeePTB documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
DPNEGF Documentation
=================================================

**DPNEGF** is a Python package that integrates the Deep Learning Tight-Binding (**DeePTB**) approach with the Non-Equilibrium Green’s Function (**NEGF**) method, establishing an efficient quantum transport simulation framework **DeePTB-NEGF** with first-principles accuracy. 

--------------
Key Features:
--------------

DeePTB contains two main components: 

1. **DeePTB-SK**: deep learning based local environment dependent Slater-Koster TB.

   - Customizable Slater-Koster parameterization with neural network corrections.
   - Flexible basis and exchange-correlation functional choices.
   - Handle systems with strong spin-orbit coupling (SOC) effects.

2. **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators.

   - Construct DFT Hamiltonians/density and overlap matrices under full LCAO basis.
   - Utilize (**S**\ trictly) **L**\ ocalized **E**\ quivariant **M**\ essage-passing (**(S)LEM**) model for high data-efficiency and accuracy.
   - Employs SO(2) convolution for efficient handling of higher-order orbitals in LCAO basis.


For more details, see our papers:

* `DeePTB-NEGF: arXiv:2411.08800 <https://arxiv.org/abs/2411.08800v2>`_
* `DeePTB-SK: Nat Commun 15, 6772 (2024) <https://doi.org/10.1038/s41467-024-51006-4>`_
* `DeePTB-E3: ICLR 2025 Spotlight <https://openreview.net/forum?id=kpq3IIjUD3>`_


.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   easy_install
   hands_on/index

.. toctree::
   :maxdepth: 2
   :caption: INPUT TAG
   
   input_params/index

.. toctree::
   :maxdepth: 2
   :caption: Citing DeePTB-NEGF

   CITATIONS

.. toctree::
   :maxdepth: 2
   :caption: Developing Team

   DevelopingTeam

.. toctree::
   :maxdepth: 2
   :caption: Community

   CONTRIBUTING

