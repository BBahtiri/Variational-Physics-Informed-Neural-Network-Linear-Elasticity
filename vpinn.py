"""
Variational Physics-Informed Neural Networks (VPINN) for 2D Linear Elasticity

This implementation solves the 2D linear elasticity problem using VPINNs as described in:
'hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition'
by Kharazmi et al.

Theory
------
1. Strong Form:
   The linear elasticity equations in strong form are:
   -∇·σ = f    in Ω      (equilibrium equation)
   u = u̅       on ΓD     (Dirichlet boundary)
   σ·n = t̅     on ΓN     (Neumann boundary)
   
   where:
   - σ: Cauchy stress tensor
   - f: Body force vector (assumed zero here)
   - u: Displacement vector field
   - n: Outward unit normal vector
   - ΓD, ΓN: Dirichlet and Neumann boundaries

2. Constitutive Relations:
   For linear elastic isotropic materials:
   σ = 2μ ε + λ tr(ε)I
   
   where:
   - μ = E/(2(1+ν)): First Lamé parameter (shear modulus)
   - λ = Eν/((1+ν)(1-2ν)): Second Lamé parameter
   - ε: Small strain tensor
   - I: Identity tensor
   - E: Young's modulus
   - ν: Poisson's ratio

3. Strain-Displacement Relations:
   ε = 1/2(∇u + ∇uᵀ)
   
   Components:
   εxx = ∂u/∂x
   εyy = ∂v/∂y
   εxy = 1/2(∂u/∂y + ∂v/∂x)

4. Weak Form:
   ∫Ω σ(u):ε(v) dΩ = ∫Ω f·v dΩ + ∫ΓN t̅·v dS
   
   where v are test functions of the form:
   νn(x,y) = (Pn+1(x) - Pn-1(x))(Pn+1(y) - Pn-1(y))
   
   with Pn being Legendre polynomials.

5. Neural Network Approximation:
   The displacement field is approximated as:
   u_h(x) = g(x) + d(x)·N(x)
   
   where:
   - g(x): Boundary extension function
   - d(x): Distance function
   - N(x): Neural network output

Implementation Details
--------------------
1. Network Architecture:
   - Separate networks for u and v displacements
   - Each network takes only its relevant coordinate
   - Deep neural networks with tanh activation

2. Loss Function:
   Total loss = Variational loss + Boundary loss
   where:
   - Variational loss: Weak form residual
   - Boundary loss: Penalty on boundary conditions

3. Integration:
   - Gauss-Legendre quadrature for domain integrals
   - Transformed coordinates from [-1,1] to [0,1]

4. Test Functions:
   - Based on Legendre polynomials
   - Pre-computed at quadrature points
   - Derivatives computed analytically

5. Test Functions and Integration
   ----------------------------
   a) Test Function Construction:
      The test functions are constructed as tensor products of 1D Legendre-based functions:
      νn(x,y) = ψi(x)ψj(y)
      where ψk(t) = Pk+1(t) - Pk-1(t)
      
      This choice of test functions has several advantages:
      - Orthogonality properties of Legendre polynomials
      - Natural handling of higher-order derivatives
      - Good approximation properties for elliptic problems
   
   b) Numerical Integration:
      We use Gauss-Legendre quadrature for numerical integration:
      ∫Ω f(x) dx ≈ Σᵢ wᵢf(xᵢ)
      
      The quadrature points xᵢ and weights wᵢ are chosen to exactly integrate
      polynomials up to degree 2n-1, where n is the number of points.

   c) Domain Transformation:
      The problem involves two coordinate transformations:
      
      1. Reference to Physical Domain:
         - Reference domain: [-1,1]×[-1,1] (where Legendre polynomials are defined)
         - Physical domain: [0,1]×[0,1] (actual problem domain)
         
         Transformation: x = (ξ + 1)/2, y = (η + 1)/2
         where (ξ,η) ∈ [-1,1]×[-1,1] and (x,y) ∈ [0,1]×[0,1]
      
      2. Jacobian of Transformation:
         - dx/dξ = dy/dη = 1/2 (from the linear mapping)
         - Jacobian determinant = 1/4
         
         This affects both:
         - Integration weights: w → w/4
         - Derivatives: ∂/∂x → 2∂/∂ξ
      
      The transformations are needed because:
      1. Legendre polynomials are naturally defined on [-1,1]
      2. Quadrature rules are typically given for [-1,1]
      3. The problem domain is [0,1]×[0,1]

   d) Implementation Details:
      1. Test Function Computation:
         - Pre-compute test functions at quadrature points
         - Transform derivatives according to chain rule
         - Store in efficient tensor format
      
      2. Quadrature:
         - Use high-order Gauss-Legendre quadrature
         - Scale weights by Jacobian determinant
         - Transform quadrature points to physical domain
      
      3. Variational Form:
         The weak form must account for the coordinate transformation:
         ∫[0,1]×[0,1] σ:ε dxdy = ∫[-1,1]×[-1,1] σ:ε |J| dξdη
         
         where |J| = 1/4 is the Jacobian determinant

Example:
--------
For a test function ν(x,y) and its derivatives:

1. Original test function:
   ν(x,y) = (P₂(x) - P₀(x))(P₂(y) - P₀(y))

2. Coordinate transformation:
   x = (ξ + 1)/2, y = (η + 1)/2

3. Chain rule for derivatives:
   ∂ν/∂x = 2∂ν/∂ξ
   ∂ν/∂y = 2∂ν/∂η

4. Integration:
   ∫[0,1]×[0,1] f(x,y) dxdy = (1/4)∫[-1,1]×[-1,1] f((ξ+1)/2, (η+1)/2) dξdη
                              ≈ (1/4)Σᵢⱼ wᵢwⱼf((ξᵢ+1)/2, (ηⱼ+1)/2)
"""

import torch
import numpy as np
from typing import Tuple
from scipy.special import legendre
from config import CONFIG
from models import ElasticityNet
from FE.basis_2d_QN_Legendre_Special import Basis2DQNLegendreSpecial
from FE.quadratureformulas_quad2d import Quadratureformulas_Quad2D

class VPINN:
    def __init__(self):
        """
        Initialize VPINN solver with material properties, neural networks, and numerical parameters.
        
        Material Properties
        ------------------
        E: Young's modulus
           Measure of material stiffness
        ν: Poisson's ratio
           Measure of lateral contraction
        
        Lamé Parameters
        --------------
        λ = (E·ν)/((1+ν)(1-2ν))
        μ = E/(2(1+ν))
        
        Network Architecture
        -------------------
        - Input: (x,y) coordinates
        - Hidden layers: Multiple layers with tanh activation
        - Output: (u,v) displacements
        """
        # Define material properties for linear elastic material
        self.E = CONFIG['YOUNGS_MODULUS']
        self.nu = CONFIG['POISSON_RATIO']
        
        # Calculate Lamé parameters for plane strain
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        
        # Calculate plane strain modulus
        self.E_prime = self.E / (1 - self.nu**2)  # Plane strain modulus
        
        # Initialize neural networks for displacement prediction
        self.network = ElasticityNet(
            num_hidden_layers=CONFIG['NUM_HIDDEN_LAYERS'],
            neurons_per_layer=CONFIG['NEURONS_PER_LAYER']
        )
        
        # Setup optimizers
        self.adam_optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=CONFIG['LEARNING_RATE']
        )
        
        # Add back the scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.adam_optimizer,
            step_size=CONFIG['LR_DECAY_STEPS'],
            gamma=CONFIG['LR_DECAY_RATE']
        )
        
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.network.parameters(),
            lr=0.1,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        
        # Initialize basis functions using Legendre Special polynomials
        num_test = CONFIG['TEST_FUNCTION_END'] - CONFIG['TEST_FUNCTION_START']
        self.basis = Basis2DQNLegendreSpecial(num_test**2)
        
        # Initialize quadrature using Gauss-Legendre
        self.quad = Quadratureformulas_Quad2D(
            quad_order=CONFIG['NUM_QUAD_POINTS'],
            quad_type="gauss-legendre"
        )
        
        # Get quadrature points and weights
        self.quad_weights, self.xi_quad, self.eta_quad = self.quad.get_quad_values()
        self.quad_weights = torch.tensor(self.quad_weights, dtype=torch.float32)
        
        # Pre-generate integration points (already in [-1,1] domain)
        self.x_quad = torch.stack([
            torch.tensor(self.xi_quad, dtype=torch.float32),
            torch.tensor(self.eta_quad, dtype=torch.float32)
        ], dim=1)
        
        # Transform quadrature points to [0,1] domain for boundary points
        x_quad_01 = (self.x_quad + 1) / 2
        
        # Pre-generate boundary points
        num_boundary = CONFIG['NUM_BOUNDARY_POINTS']
        self.x_boundary = torch.cat([
            torch.cartesian_prod(torch.tensor([0.0]), torch.linspace(0, 1, num_boundary)),
            torch.cartesian_prod(torch.tensor([1.0]), torch.linspace(0, 1, num_boundary)),
            torch.cartesian_prod(torch.linspace(0, 1, num_boundary), torch.tensor([0.0])),
        ])
        
        # Pre-compute boundary masks
        self.left_mask = self.x_boundary[:, 0] < 1e-5
        self.bottom_mask = self.x_boundary[:, 1] < 1e-5
        self.right_mask = self.x_boundary[:, 0] > 0.99
        
        # Pre-compute test functions
        self.test_functions = {}
        for n in range(CONFIG['TEST_FUNCTION_START'], CONFIG['TEST_FUNCTION_END']):
            self.test_functions[n] = self.precompute_test_function(n)
    
    def get_strain(self, u: torch.Tensor, x: torch.Tensor, 
                   create_graph: bool = True) -> torch.Tensor:
        """
        Calculate strain tensor components using automatic differentiation.
        
        The strain tensor for small deformations is:
        ε = [εxx  εxy]
            [εxy  εyy]
        
        where:
        εxx = ∂u/∂x
        εyy = ∂v/∂y
        εxy = 1/2(∂u/∂y + ∂v/∂x)
        
        Parameters
        ----------
        u : torch.Tensor
            Displacement field tensor [batch_size, 2]
        x : torch.Tensor
            Coordinate points tensor [batch_size, 2]
        create_graph : bool
            Flag for creating computation graph for higher derivatives
        
        Returns
        -------
        torch.Tensor
            Strain components [εxx, εyy, εxy]
        """
        # Calculate displacement gradients
        grad_u = torch.autograd.grad(
            u[:, 0].sum(), x, create_graph=create_graph, retain_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            u[:, 1].sum(), x, create_graph=create_graph, retain_graph=True
        )[0]
        
        # Small strain tensor
        epsilon_xx = grad_u[:, 0]
        epsilon_yy = grad_v[:, 1]
        epsilon_xy = 0.5 * (grad_u[:, 1] + grad_v[:, 0])
        
        return torch.stack([epsilon_xx, epsilon_yy, epsilon_xy], dim=1)
    
    def get_stress(self, strain: torch.Tensor) -> torch.Tensor:
        """
        Calculate stress tensor components using Hooke's law for plane strain.
        
        For plane strain conditions:
        σxx = (λ + 2μ)εxx + λεyy
        σyy = λεxx + (λ + 2μ)εyy
        σxy = 2μεxy
        
        Parameters
        ----------
        strain : torch.Tensor
            Strain components [εxx, εyy, εxy]
        
        Returns
        -------
        torch.Tensor
            Stress components [σxx, σyy, σxy]
        """
        epsilon_xx, epsilon_yy, epsilon_xy = strain[:, 0], strain[:, 1], strain[:, 2]
        
        # Plane strain stress-strain relationship
        sigma_xx = self.E_prime * (epsilon_xx + self.nu * epsilon_yy)
        sigma_yy = self.E_prime * (self.nu * epsilon_xx + epsilon_yy)
        sigma_xy = self.mu * epsilon_xy
        
        return torch.stack([sigma_xx, sigma_yy, sigma_xy], dim=1)
    
    def precompute_test_function(self, index: int):
        """Pre-compute test functions and their derivatives at quadrature points."""
        xi = self.xi_quad
        eta = self.eta_quad
        
        # Get the test function and its derivatives at quadrature points
        # The special basis class uses different test function formulation
        values = self.basis.value(xi, eta)
        grad_x = self.basis.gradx(xi, eta)
        grad_y = self.basis.grady(xi, eta)
        
        # Extract the specific test function (index-1 because test functions start from 1)
        test_value = torch.tensor(values[index-1], dtype=torch.float32)
        test_grad_x = torch.tensor(grad_x[index-1], dtype=torch.float32)
        test_grad_y = torch.tensor(grad_y[index-1], dtype=torch.float32)
        
        # Stack gradients into tensor of shape [num_quad_points, 2]
        test_grad = torch.stack([test_grad_x, test_grad_y], dim=1)
        
        return test_value, test_grad
    
    def variational_loss(self, x_quad: torch.Tensor) -> torch.Tensor:
        """
        Calculate the variational form loss using the weak formulation.
        
        The weak form is:
        ∫Ω σ(u):ε(v) dΩ = ∫Ω f·v dΩ + ∫ΓN t̅·v dS
        
        Discretized using Gauss quadrature:
        Σᵢ wᵢ[σ(u(xᵢ)):ε(v(xᵢ))] ≈ Σᵢ wᵢ[f(xᵢ)·v(xᵢ)] + boundary terms
        
        Parameters
        ----------
        x_quad : torch.Tensor
            Quadrature points for numerical integration
        
        Returns
        -------
        torch.Tensor
            Scalar loss value from weak form residual
        """
        x_quad = x_quad.detach().requires_grad_(True)
        
        # Get network prediction
        u_pred = self.network(x_quad)
        strain = self.get_strain(u_pred, x_quad)
        stress = self.get_stress(strain)
        
        # Scale quadrature weights
        jacobian = 0.5 * 0.5
        wx = self.quad_weights * jacobian
        
        total_residual = 0.0
        num_test = CONFIG['TEST_FUNCTION_END'] - CONFIG['TEST_FUNCTION_START']
        
        for n in range(CONFIG['TEST_FUNCTION_START'], CONFIG['TEST_FUNCTION_END']):
            _, test_grad = self.test_functions[n]
            
            # Scale gradients by Jacobian
            test_grad = test_grad * 2.0
            
            # Compute weak form integrand: σ:ε(ν)
            integrand = (
                stress[:, 0] * test_grad[:, 0] +    # σxx * ∂νx/∂x
                stress[:, 1] * test_grad[:, 1] +    # σyy * ∂νy/∂y
                stress[:, 2] * (                    # σxy * (∂νx/∂y + ∂νy/∂x)
                    test_grad[:, 1] + test_grad[:, 0]
                )
            )
            
            # Compute residual for this test function
            R_n = torch.sum(integrand * wx)
            total_residual += torch.abs(R_n)
        
        return total_residual / num_test
    
    def boundary_loss(self, x_boundary: torch.Tensor) -> torch.Tensor:
        """Calculate loss for boundary conditions using pre-computed masks."""
        u_pred = self.network(self.x_boundary)
        
        # Use pre-computed masks
        left_loss = torch.mean(u_pred[self.left_mask, 0]**2)
        bottom_loss = torch.mean(u_pred[self.bottom_mask, 1]**2)
        right_loss = torch.mean((u_pred[self.right_mask, 0] - CONFIG['DISPLACEMENT_X'])**2)
        
        return left_loss + bottom_loss + right_loss
    
    def train_step_adam(self) -> Tuple[float, float]:
        """Perform one training step using Adam optimizer."""
        self.adam_optimizer.zero_grad()
        
        v_loss = self.variational_loss(self.x_quad)
        b_loss = self.boundary_loss(self.x_boundary)
        total_loss = v_loss + CONFIG['BOUNDARY_WEIGHT'] * b_loss
        
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 
            CONFIG['GRADIENT_CLIP']
        )
        
        self.adam_optimizer.step()
        
        return v_loss.item(), b_loss.item()