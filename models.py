import torch
import torch.nn as nn
from config import CONFIG

class DisplacementNet(nn.Module):
    """Neural network for a single displacement component."""
    def __init__(self, num_hidden_layers: int = 6, neurons_per_layer: int = 30):
        super().__init__()
        self.input_layer = nn.Linear(2, neurons_per_layer)
        
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.extend([
                nn.Tanh(),
                nn.Linear(neurons_per_layer, neurons_per_layer),
            ])
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layer = nn.Linear(neurons_per_layer, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(x)
        z = self.hidden_layers(z)
        return self.output_layer(z)

class ElasticityNet(nn.Module):
    """Network architecture with separate networks for u and v displacements."""
    def __init__(self, num_hidden_layers: int = 6, neurons_per_layer: int = 30):
        super().__init__()
        
        # Separate networks for u and v
        self.u_net = DisplacementNet(num_hidden_layers, neurons_per_layer)
        self.v_net = DisplacementNet(num_hidden_layers, neurons_per_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x.clone()
        
        # Get raw network outputs using only respective coordinates
        w_u = self.u_net(x_input).squeeze(-1)  # Only x-coordinate
        w_v = self.v_net(x_input).squeeze(-1)  # Only y-coordinate
        
        # Define phi_u(x) for u (only x-dependence needed)
        alpha = CONFIG['ADF_ALPHA']
        gamma = CONFIG['ADF_GAMMA']
        phi_u = (x_input[:, 0] ** alpha) * ((1 - x_input[:, 0]) ** gamma)
        
        # Define phi_v(y) for v (only y-dependence needed)
        beta_v = CONFIG['ADF_BETA_V']
        phi_v = (x_input[:, 1] ** beta_v)  # Goes to 0 at y=0
        
        # Define boundary extension functions
        delta = CONFIG['GAMMA_DELTA']
        u_bc_right = CONFIG['DISPLACEMENT_X']
        g_u = u_bc_right * (x_input[:, 0] ** delta)  # Extension of right BC
        g_v = torch.zeros_like(x_input[:, 1])        # v=0 on bottom boundary
        
        # Compute the modified displacements
        u = g_u + phi_u * w_u
        v = g_v + phi_v * w_v
        
        return torch.stack([u, v], dim=1)