import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from scipy.special import legendre
from matplotlib.patches import Rectangle
from config import CONFIG
from models import ElasticityNet
from vpinn import VPINN

def train_vpinn(use_lbfgs: bool = False):
    """Train the VPINN model using two-stage optimization."""
    vpinn = VPINN()
    best_loss = float('inf')
    best_state = None
    
    # Stage 1: Adam optimization
    print("Stage 1: Training with Adam")
    for epoch in range(CONFIG['NUM_EPOCHS']):
        v_loss, b_loss = vpinn.train_step_adam()
        total_loss = v_loss + CONFIG['BOUNDARY_WEIGHT'] * b_loss
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_state = {k: v.cpu().clone() for k, v in vpinn.network.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Variational Loss = {v_loss:.6f}, "
                  f"Boundary Loss = {b_loss:.6f}")
    
    # Load best state from Adam
    vpinn.network.load_state_dict(best_state)
    
    # Stage 2: L-BFGS optimization
    if use_lbfgs:
        print("\nStage 2: Training with L-BFGS")
        prev_loss = float('inf')
        no_improvement_count = 0
        
        def closure():
            vpinn.lbfgs_optimizer.zero_grad()
            v_loss = vpinn.variational_loss(vpinn.x_quad)
            b_loss = vpinn.boundary_loss(vpinn.x_boundary)
            total_loss = v_loss + CONFIG['BOUNDARY_WEIGHT'] * b_loss
            total_loss.backward()
            return total_loss
        
        for i in range(CONFIG['LBFGS_MAX_ITER']):
            loss = vpinn.lbfgs_optimizer.step(closure)
            if i % 1 == 0:
                print(f"L-BFGS iteration {i}: Loss = {loss.item():.6f}")
            
            # Early stopping if loss isn't improving
            if loss.item() >= prev_loss:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                prev_loss = loss.item()
            
            if no_improvement_count > 50:  # Stop if no improvement for 50 iterations
                print("\nStopping L-BFGS: No improvement in loss")
                break
            
            if loss < 1e-4:
                print("\nConverged!")
                break
    
    return vpinn

def plot_deformation(vpinn: VPINN, num_points: int = 100):
    """Plot the deformed and undeformed configurations with displacement colormaps."""
    try:
        plt.clf()  # Clear current figure
        plt.close('all')  # Close all figures
        
        # Create visualization grid
        x = torch.linspace(0, 1, num_points)
        y = torch.linspace(0, 1, num_points)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Predict displacements
        with torch.no_grad():
            displacements = vpinn.network(xy)
        
        # Reshape predictions to grid format
        u = displacements[:, 0].reshape(num_points, num_points)
        v = displacements[:, 1].reshape(num_points, num_points)
        
        # Calculate deformed coordinates
        X_def = X + u
        Y_def = Y + v
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Undeformed configuration
        ax1.scatter(X, Y, c='lightgray', s=1, label='Undeformed')
        ax1.add_patch(Rectangle((0, 0), 1, 1, fill=False, color='gray'))
        ax1.set_title('Reference Configuration')
        ax1.set_aspect('equal')
        ax1.grid(True)
        
        # Plot 2: x-displacement
        im2 = ax2.scatter(X_def, Y_def, c=u, cmap='viridis', s=1)
        ax2.set_title('Deformed Configuration\nX-displacement')
        ax2.set_aspect('equal')
        ax2.grid(True)
        plt.colorbar(im2, ax=ax2, label='u [units]')
        
        # Plot 3: y-displacement
        im3 = ax3.scatter(X_def, Y_def, c=v, cmap='viridis', s=1)
        ax3.set_title('Deformed Configuration\nY-displacement')
        ax3.set_aspect('equal')
        ax3.grid(True)
        plt.colorbar(im3, ax=ax3, label='v [units]')
        
        # Plot 4: Total displacement magnitude
        total_disp = torch.sqrt(u**2 + v**2)
        im4 = ax4.scatter(X_def, Y_def, c=total_disp, cmap='viridis', s=1)
        ax4.set_title('Deformed Configuration\nTotal displacement')
        ax4.set_aspect('equal')
        ax4.grid(True)
        plt.colorbar(im4, ax=ax4, label='|u| [units]')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('deformation_plots.png', dpi=300, bbox_inches='tight')
        plt.close('all')  # Close all figures again after saving
        
        print("\nPlots have been saved to 'deformation_plots.png'")
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        plt.close('all')  # Ensure figures are closed even if error occurs

def plot_comparison(vpinn: VPINN):
    num_points = 100
    x = torch.linspace(0, 1, num_points)
    y = torch.linspace(0, 1, num_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # VPINN solution
    with torch.no_grad():
        displacements = vpinn.network(xy)
    u_vpinn = displacements[:, 0].reshape(num_points, num_points)
    v_vpinn = displacements[:, 1].reshape(num_points, num_points)
    
    # Analytical solution
    u_analytical = 0.1 * X
    v_analytical = -vpinn.nu * 0.1 * Y
    
    # Calculate error
    u_error = torch.abs(u_vpinn - u_analytical)
    v_error = torch.abs(v_vpinn - v_analytical)
    
    print(f"Maximum u error: {u_error.max():.6f}")
    print(f"Maximum v error: {v_error.max():.6f}")
    print(f"Average u error: {u_error.mean():.6f}")
    print(f"Average v error: {v_error.mean():.6f}")


def plot_stress_strain(vpinn: VPINN, num_points: int = 100):
    """Plot stress and strain distributions."""
    try:
        plt.clf()  # Clear current figure
        plt.close('all')  # Close all figures
        
        # Create visualization grid
        x = torch.linspace(0, 1, num_points)
        y = torch.linspace(0, 1, num_points)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Important: Move this outside of no_grad context
        xy.requires_grad_(True)
        u_pred = vpinn.network(xy)
        
        # Compute strain and stress
        strain = vpinn.get_strain(u_pred, xy, create_graph=False)
        
        # Now we can use no_grad for the stress computation
        with torch.no_grad():
            stress = vpinn.get_stress(strain)
            
            # Detach and reshape components
            strain = strain.detach().reshape(num_points, num_points, -1)
            stress = stress.detach().reshape(num_points, num_points, -1)
            
            # Create figure with 3x2 subplots
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 20))
            
            # Plot strain components
            im1 = ax1.contourf(X.numpy(), Y.numpy(), strain[..., 0].numpy(), levels=20, cmap='viridis')
            ax1.set_title('εxx')
            plt.colorbar(im1, ax=ax1)
            ax1.set_aspect('equal')
            ax1.grid(True)
            
            im2 = ax2.contourf(X.numpy(), Y.numpy(), strain[..., 1].numpy(), levels=20, cmap='viridis')
            ax2.set_title('εyy')
            plt.colorbar(im2, ax=ax2)
            ax2.set_aspect('equal')
            ax2.grid(True)
            
            im3 = ax3.contourf(X.numpy(), Y.numpy(), strain[..., 2].numpy(), levels=20, cmap='viridis')
            ax3.set_title('εxy')
            plt.colorbar(im3, ax=ax3)
            ax3.set_aspect('equal')
            ax3.grid(True)
            
            # Plot stress components
            im4 = ax4.contourf(X.numpy(), Y.numpy(), stress[..., 0].numpy(), levels=20, cmap='viridis')
            ax4.set_title('σxx')
            plt.colorbar(im4, ax=ax4)
            ax4.set_aspect('equal')
            ax4.grid(True)
            
            im5 = ax5.contourf(X.numpy(), Y.numpy(), stress[..., 1].numpy(), levels=20, cmap='viridis')
            ax5.set_title('σyy')
            plt.colorbar(im5, ax=ax5)
            ax5.set_aspect('equal')
            ax5.grid(True)
            
            im6 = ax6.contourf(X.numpy(), Y.numpy(), stress[..., 2].numpy(), levels=20, cmap='viridis')
            ax6.set_title('σxy')
            plt.colorbar(im6, ax=ax6)
            ax6.set_aspect('equal')
            ax6.grid(True)
            
            # Add titles and adjust layout
            fig.suptitle('Strain Components (top) and Stress Components (bottom)', fontsize=16)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('stress_strain_plots.png', dpi=300, bbox_inches='tight')
            plt.close('all')  # Close all figures again after saving
            
            print("\nStress-strain plots have been saved to 'stress_strain_plots.png'")
            
    except Exception as e:
        print(f"Error in plotting stress-strain: {str(e)}")
        plt.close('all')  # Ensure figures are closed even if error occurs

def plot_geometry_and_points(vpinn: VPINN):
    """Plot the geometry with quadrature and boundary points."""
    try:
        plt.clf()  # Clear current figure
        plt.close('all')  # Close all figures
        
        plt.figure(figsize=(10, 10))
        
        # Plot domain boundary
        plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2, label='Domain')
        
        # Plot quadrature points (transform from [-1,1] to [0,1] for plotting)
        quad_points = (vpinn.x_quad + 1) / 2
        plt.scatter(quad_points[:, 0], quad_points[:, 1], 
                   c='blue', s=20, alpha=0.5, label='Quadrature points')
        
        # Plot boundary points
        plt.scatter(vpinn.x_boundary[vpinn.left_mask, 0], 
                   vpinn.x_boundary[vpinn.left_mask, 1], 
                   c='red', s=50, label='Left boundary')
        
        plt.scatter(vpinn.x_boundary[vpinn.right_mask, 0], 
                   vpinn.x_boundary[vpinn.right_mask, 1], 
                   c='green', s=50, label='Right boundary')
        
        plt.scatter(vpinn.x_boundary[vpinn.bottom_mask, 0], 
                   vpinn.x_boundary[vpinn.bottom_mask, 1], 
                   c='orange', s=50, label='Bottom boundary')
        
        plt.title('Domain with Quadrature and Boundary Points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('geometry_points.png', dpi=300, bbox_inches='tight')
        plt.close('all')  # Close all figures again after saving
        
        print("\nGeometry plot has been saved to 'geometry_points.png'")
        
    except Exception as e:
        print(f"Error in plotting geometry: {str(e)}")
        plt.close('all')  # Ensure figures are closed even if error occurs

def save_vpinn_results(vpinn: VPINN, num_points: int = 100):
    """Save VPINN inputs and outputs to both .npz and .txt files."""
    # Create regular grid of points
    x = torch.linspace(0, 1, num_points)
    y = torch.linspace(0, 1, num_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Get VPINN predictions
    with torch.no_grad():
        displacements = vpinn.network(xy)
    u = displacements[:, 0].reshape(num_points, num_points)
    v = displacements[:, 1].reshape(num_points, num_points)
    
    # Calculate analytical solution
    u_analytical = 0.1 * X
    v_analytical = -vpinn.nu * 0.1 * Y
    
    # Calculate strains and stresses
    xy.requires_grad_(True)
    u_pred = vpinn.network(xy)
    strain = vpinn.get_strain(u_pred, xy, create_graph=False)
    stress = vpinn.get_stress(strain)
    
    # Reshape results
    strain = strain.detach().reshape(num_points, num_points, -1)
    stress = stress.detach().reshape(num_points, num_points, -1)
    
    # Save results to numpy files
    np.savez('vpinn_results.npz',
             x=X.numpy(), y=Y.numpy(),
             u_pred=u.numpy(), v_pred=v.numpy(),
             u_analytical=u_analytical.numpy(), v_analytical=v_analytical.numpy(),
             eps_xx=strain[..., 0].numpy(), eps_yy=strain[..., 1].numpy(), eps_xy=strain[..., 2].numpy(),
             sig_xx=stress[..., 0].numpy(), sig_yy=stress[..., 1].numpy(), sig_xy=stress[..., 2].numpy())
    
    # Save as .txt with headers
    # Flatten and combine all results
    results_array = np.column_stack([
        X.flatten().numpy(),
        Y.flatten().numpy(),
        u.flatten().numpy(),
        v.flatten().numpy(),
        u_analytical.flatten().numpy(),
        v_analytical.flatten().numpy(),
        strain[..., 0].flatten().numpy(),
        strain[..., 1].flatten().numpy(),
        strain[..., 2].flatten().numpy(),
        stress[..., 0].flatten().numpy(),
        stress[..., 1].flatten().numpy(),
        stress[..., 2].flatten().numpy()
    ])
    
    # Create header for the text file
    header = "x y u_pred v_pred u_analytical v_analytical eps_xx eps_yy eps_xy sig_xx sig_yy sig_xy"
    
    # Save to text file
    np.savetxt('vpinn_results.txt', results_array, 
               header=header,
               delimiter=' ',
               fmt='%.6e')  # Scientific notation with 6 decimal places
    
    print("\nResults have been saved to:")
    print("1. 'vpinn_results.npz' (Binary format)")
    print("2. 'vpinn_results.txt' (Text format)")
    print("\nText file format:")
    print("Columns:", header)

def analytical_stress_strain(x: torch.Tensor, y: torch.Tensor, E: float, nu: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate analytical stress and strain for uniaxial tension."""
    # Strain components
    eps_xx = 0.1  # εxx = 0.1 (constant throughout domain)
    eps_yy = -nu * eps_xx  # εyy = -ν * εxx (Poisson effect)
    eps_xy = 0.0  # No shear strain
    
    # Stress components (plane strain)
    sigma_xx = E * (1 - nu) / ((1 + nu) * (1 - 2*nu)) * eps_xx  # σxx
    sigma_yy = E * nu / ((1 + nu) * (1 - 2*nu)) * eps_xx  # σyy
    sigma_xy = 0.0  # No shear stress
    
    strain = torch.stack([
        torch.full_like(x, eps_xx),
        torch.full_like(x, eps_yy),
        torch.full_like(x, eps_xy)
    ], dim=-1)
    
    stress = torch.stack([
        torch.full_like(x, sigma_xx),
        torch.full_like(x, sigma_yy),
        torch.full_like(x, sigma_xy)
    ], dim=-1)
    
    return strain, stress

def plot_stress_strain_comparison(vpinn: VPINN, num_points: int = 100):
    """Plot comparison of predicted and analytical stress-strain distributions."""
    # Create grid
    x = torch.linspace(0, 1, num_points)
    y = torch.linspace(0, 1, num_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Get VPINN predictions
    xy.requires_grad_(True)
    u_pred = vpinn.network(xy)
    strain_pred = vpinn.get_strain(u_pred, xy)
    stress_pred = vpinn.get_stress(strain_pred)
    
    # Detach tensors and convert to numpy
    strain_pred = strain_pred.detach()
    stress_pred = stress_pred.detach()
    
    # Get analytical solutions
    strain_anal, stress_anal = analytical_stress_strain(X.flatten(), Y.flatten(), 
                                                      vpinn.E, vpinn.nu)
    
    # Calculate errors
    strain_error = torch.abs(strain_pred - strain_anal)
    stress_error = torch.abs(stress_pred - stress_anal)
    
    # Reshape for plotting
    components = ['xx', 'yy', 'xy']
    titles_strain = [f'ε_{c}' for c in components]
    titles_stress = [f'σ_{c}' for c in components]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Plot strains
    for i, (title, sp, sa, se) in enumerate(zip(titles_strain, 
                                               strain_pred.reshape(-1, 3).T,
                                               strain_anal.reshape(-1, 3).T,
                                               strain_error.reshape(-1, 3).T)):
        # Convert to numpy and reshape to 2D grid
        sp_np = sp.reshape(num_points, num_points).numpy()
        sa_np = sa.reshape(num_points, num_points).numpy()
        se_np = se.reshape(num_points, num_points).numpy()
        
        # Predicted
        im = axes[i,0].contourf(X.numpy(), Y.numpy(), sp_np, levels=20, cmap='viridis')
        axes[i,0].set_title(f'{title} (Predicted)')
        plt.colorbar(im, ax=axes[i,0])
        
        # Analytical
        im = axes[i,1].contourf(X.numpy(), Y.numpy(), sa_np, levels=20, cmap='viridis')
        axes[i,1].set_title(f'{title} (Analytical)')
        plt.colorbar(im, ax=axes[i,1])
        
        # Error
        im = axes[i,2].contourf(X.numpy(), Y.numpy(), se_np, levels=20, cmap='viridis')
        axes[i,2].set_title(f'{title} (Error)')
        plt.colorbar(im, ax=axes[i,2])
    
    plt.tight_layout()
    plt.savefig('strain_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot stresses
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, (title, sp, sa, se) in enumerate(zip(titles_stress,
                                               stress_pred.reshape(-1, 3).T,
                                               stress_anal.reshape(-1, 3).T,
                                               stress_error.reshape(-1, 3).T)):
        # Convert to numpy and reshape to 2D grid
        sp_np = sp.reshape(num_points, num_points).numpy()
        sa_np = sa.reshape(num_points, num_points).numpy()
        se_np = se.reshape(num_points, num_points).numpy()
        
        # Rest of plotting code remains the same
        im = axes[i,0].contourf(X.numpy(), Y.numpy(), sp_np, levels=20, cmap='viridis')
        axes[i,0].set_title(f'{title} (Predicted)')
        plt.colorbar(im, ax=axes[i,0])
        
        im = axes[i,1].contourf(X.numpy(), Y.numpy(), sa_np, levels=20, cmap='viridis')
        axes[i,1].set_title(f'{title} (Analytical)')
        plt.colorbar(im, ax=axes[i,1])
        
        im = axes[i,2].contourf(X.numpy(), Y.numpy(), se_np, levels=20, cmap='viridis')
        axes[i,2].set_title(f'{title} (Error)')
        plt.colorbar(im, ax=axes[i,2])
    
    plt.tight_layout()
    plt.savefig('stress_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nStrain Error Statistics:")
    for i, comp in enumerate(components):
        print(f"\nε_{comp}:")
        print(f"Maximum error: {strain_error[..., i].max().item():.6f}")
        print(f"Average error: {strain_error[..., i].mean().item():.6f}")
    
    print("\nStress Error Statistics:")
    for i, comp in enumerate(components):
        print(f"\nσ_{comp}:")
        print(f"Maximum error: {stress_error[..., i].max().item():.6f}")
        print(f"Average error: {stress_error[..., i].mean().item():.6f}")

if __name__ == "__main__":
    vpinn = VPINN()
    plot_geometry_and_points(vpinn)
    trained_vpinn = train_vpinn(use_lbfgs=CONFIG['USE_LBFGS'])
    plot_deformation(trained_vpinn)
    plot_stress_strain(trained_vpinn)
    plot_comparison(trained_vpinn)
    save_vpinn_results(trained_vpinn)
    plot_stress_strain_comparison(trained_vpinn)
    
