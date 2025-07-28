# Variational Physics-Informed Neural Network (VPINN) for 2D Elasticity

## 1. Overview

This project implements a **Variational Physics-Informed Neural Network (VPINN)** to solve a 2D linear elasticity problem. The specific case is a **square plate under uniaxial tension**, modeled using **plane stress** assumptions.

The methodology is inspired by the *hp-VPINNs* paper by Kharazmi et al., which leverages the calculus of variations to formulate the loss function. The network is trained to minimize the residual of the weak form of the governing PDE.

- **Reference**: Kharazmi, E., Zhang, Z., & Karniadakis, G. E. (2021). [*hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition*](https://www.sciencedirect.com/science/article/pii/S0045782520307325).

## 2. Problem Formulation

### Domain & Boundary Conditions
- **Domain**: A unit square plate, $\Omega = [0,1] \times [0,1]$.
- **Boundary Conditions (BCs)**:
  - **Left Edge** ($x=0$): Fixed horizontally, $u=0$.
  - **Bottom Edge** ($y=0$): Fixed vertically, $v=0$.
  - **Right Edge** ($x=1$): Prescribed horizontal displacement, $u=0.1$.
  - **Top Edge** ($y=1$): Free (Neumann BC).

### Physical Model
The system is governed by the equilibrium equation (assuming no body forces):
$$\nabla \cdot \boldsymbol{\sigma} = \mathbf{0} \quad \text{in } \Omega$$
The model assumes a linear elastic, isotropic material under **plane stress** conditions. The stress-strain relationship is given by Hooke's Law.

## 3. VPINN Methodology

### Variational (Weak) Form
Instead of minimizing the strong form residual ($|\nabla \cdot \boldsymbol{\sigma}|^2$), the VPINN minimizes the energy functional described by the weak form. The loss function is derived from the principle of virtual work:
```math
\mathcal{L}_{variational} = \sum_{n} \left( \int_{\Omega} \boldsymbol{\sigma}(\mathbf{u}_{NN}) : \boldsymbol{\epsilon}(\mathbf{v}_n) \,d\Omega \right)^2
```
where $\mathbf{u}_{NN}$ is the displacement field predicted by the neural network and $\mathbf{v}_n$ are vector test functions.

### Implementation Details
- **Test Functions**: The test functions $\mathbf{v}_n$ are constructed from Legendre polynomials of the form
```math
\psi_k(t)=P_{k+1}(t)-P_{k-1}(t)$
```
, which are well-suited for variational methods.
- **Numerical Integration**: The integral in the weak form is approximated numerically using **Gauss-Legendre quadrature**.
- **Coordinate Transformation**: The network operates on the physical domain `[0,1]x[0,1]`, but the Legendre polynomials and quadrature rules are defined on the reference domain `[-1,1]x[-1,1]`. A linear mapping is used, and the change of variables is handled by applying a **Jacobian correction** to the integration weights and spatial derivatives.

### Network Architecture & BC Enforcement
The Dirichlet boundary conditions are enforced analytically using an **Augmented Deep Formulation (ADF)**. The predicted displacement $\mathbf{u}_{NN}$ is not the direct output of the network but is constructed as:
```math
\mathbf{u}_{NN}(\mathbf{x}) = \mathbf{g}(\mathbf{x}) + \mathbf{\Phi}(\mathbf{x}) \odot N(\mathbf{x}; \theta)
```
- $N(\mathbf{x}; \theta)$: The raw output from the neural network.
- $\mathbf{g}(\mathbf{x})$: An extension function that satisfies the non-homogeneous BCs. For this problem, $g_u = 0.1 \cdot x^\delta$.
- $\mathbf{\Phi}(\mathbf{x})$: A distance function that is zero on the Dirichlet boundaries, ensuring the network's output does not alter the imposed conditions. Here, $\Phi_u = x^\alpha (1-x)^\gamma$ and $\Phi_v = y^\beta$.

This structure guarantees that the BCs are met exactly, regardless of the network's weights.

### Training
- **Loss Function**: The total loss combines the variational loss with an explicit (though architecturally redundant) boundary loss term to guide training: $L_{total} = L_{variational} + w \cdot L_{boundary}$.
- **Optimization**: A two-stage optimization process is used for robust convergence:
    1.  **Adam**: For rapid, broad exploration of the loss landscape.
    2.  **L-BFGS**: For fine-tuning and achieving high-precision results near the minimum.

## 4. Project Structure

- `main.py`: Main script to run the training, evaluation, and plotting.
- `vpinn.py`: Core VPINN class, handling the variational loss, boundary conditions, and optimizers.
- `models.py`: Defines the `ElasticityNet` neural network architecture with the ADF for BC enforcement.
- `config.py`: Contains all hyperparameters, material properties, and training settings.
- `FE/`: Helper module for generating basis functions and quadrature rules.

## 5. How to Use

1.  **Install dependencies**:
    ```bash
    pip install torch matplotlib scipy numpy
    ```
2.  **Run the simulation**:
    ```bash
    python main.py
    ```
3.  **Check the outputs**:
    - `deformation_plots.png`: Visualization of undeformed vs. deformed mesh and displacement components.
    - `stress_strain_plots.png`: Contour plots of all stress and strain components.
    - `strain_comparison.png` & `stress_comparison.png`: Comparison of VPINN results against the analytical solution.
    - `vpinn_results.npz` & `vpinn_results.txt`: Saved numerical data for all fields at grid points.
