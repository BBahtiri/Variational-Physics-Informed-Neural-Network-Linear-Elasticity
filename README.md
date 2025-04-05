# Variational Physics-Informed Neural Network (VPINN) for 2D Linear Elasticity

## Overview
This implementation solves plane strain linear elasticity problems using VPINNs, following the methodology from [hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition](https://www.sciencedirect.com/science/article/pii/S0045782520307325?casa_token=ADrMPLvUAQsAAAAA:4zo2-HImi9vVf2RTt-vpI0LAf6fVTAhbDoRKZEnrLdOm1GKWw2nwa_SsjirCUCe2X02qmYVK). The specific problem solved is a **square plate under uniaxial tension** with mixed boundary conditions.

## Problem Description
- **Domain**: Unit square [0,1]×[0,1]
- **Boundary Conditions**:
  - Left edge (x=0): Fixed in x-direction (u=0)
  - Bottom edge (y=0): Fixed in y-direction (v=0)
  - Right edge (x=1): Prescribed displacement in x-direction (u=0.1)
  - Top edge (y=1): Free surface (natural boundary condition)
- **Material Properties**:
  - Young's modulus (E): 1.0 GPa
  - Poisson's ratio (ν): 0.3

## Key Features
1. **VPINN Architecture**:
   - Separate neural networks for x (u) and y (v) displacements
   - Custom boundary-aware architecture with:
     - Distance functions for boundary condition enforcement
     - Adaptive activation functions
   - Two-stage optimization (Adam + L-BFGS)

2. **Weak Form Implementation**:
   - Legendre polynomial-based test functions
   - Gauss-Legendre quadrature integration
   - Automatic differentiation for strain/stress calculations
   - Plane strain constitutive relations

3. **Core Implementation**:
   ```python
   # Plane strain constitutive relationship
   σ = E/(1-ν²) * [ε_xx + νε_yy, νε_xx + ε_yy, (1-ν)/2 ε_xy]
   
   # Loss function components
   Loss = ∫(σ:ε(ν))dΩ + α(‖u_boundary‖² + ‖v_boundary‖²)
   ```
   - Mixed-variational formulation
   - Jacobian-aware coordinate transformations
   - Gradient clipping and learning rate scheduling

4. **Postprocessing**:
   - Full field displacement/stress/strain visualization
   - Error analysis against analytical solutions
   - Result export in NPZ and text formats

## Typical Outputs
- Displacement fields (u, v)
- Strain components (ε_xx, ε_yy, ε_xy)
- Stress components (σ_xx, σ_yy, σ_xy)
- Error metrics vs analytical solution
- Point-wise comparison files

## Getting Started
1. Install requirements:
   ```bash
   pip install torch matplotlib scipy numpy
   ```

2. Run main simulation:
   ```bash
   python main.py
   ```

3. Key outputs:
   - `deformation_plots.png` - Displacement visualizations
   - `stress_strain_plots.png` - Stress/strain distributions
   - `vpinn_results.npz` - Numerical results in binary format
   - `stress_comparison.png` - Validation against analytical solution

## References
- Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021). [hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition]([https://arxiv.org/abs/2104.13865](https://www.sciencedirect.com/science/article/abs/pii/S0045782520307325?casa_token=ADrMPLvUAQsAAAAA:4zo2-HImi9vVf2RTt-vpI0LAf6fVTAhbDoRKZEnrLdOm1GKWw2nwa_SsjirCUCe2X02qmYVK))
