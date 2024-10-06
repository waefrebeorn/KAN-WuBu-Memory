#adaptivekantemplate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveKANLayer(nn.Module):
    def __init__(self, input_size, output_size, num_knots=10, temperature=0.666):
        """
        Initialize an adaptive KAN layer with spline-based transformations.
        
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            num_knots (int): Number of knots in the spline function.
            temperature (float): Temperature parameter for adaptive updates.
        """
        super(AdaptiveKANLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_knots = num_knots
        self.temperature = temperature
        
        # Define spline parameters
        self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots))
        self.coeffs = nn.Parameter(torch.randn(input_size, output_size, num_knots))

    def forward(self, x):
        """
        Forward pass for the KAN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Transformed output of shape (batch_size, output_size).
        """
        weights = self.compute_spline_weights(x)
        return torch.matmul(x, weights)

    def compute_spline_weights(self, x):
        """
        Compute the spline transformation weights for input x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Spline weights of shape (input_size, output_size).
        """
        weights = F.interpolate(self.coeffs.unsqueeze(0), size=(self.num_knots,)).squeeze(0)
        return weights

    def calculate_entropy(self, logits):
        """
        Calculate entropy of the spline transformations.
        
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_classes).
            
        Returns:
            torch.Tensor: Entropy values for each class.
        """
        p = F.softmax(logits, dim=-1)
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)
        return entropy

    def adaptive_update(self, entropy, variance):
        """
        Adaptively update grid resolution and regularization based on entropy.
        
        Args:
            entropy (float): Current entropy of the spline transformations.
            variance (float): Variance of the entropy values.
        """
        if entropy < 0.1 and variance < 0.1:
            self.prune_knots()
        elif entropy > 5.0 and variance < 0.1:
            self.extend_knots()
        elif entropy < 5.0 and variance > 5.0:
            self.refine_coeffs()
        elif entropy > 5.0 and variance > 5.0:
            self.increase_capacity()
        else:
            self.moderate_update()

    def prune_knots(self):
        """Remove low-impact knots."""
        if self.num_knots > 3:  # Ensure a minimum number of knots
            self.num_knots -= 1
            self.knots = nn.Parameter(torch.linspace(-1, 1, self.num_knots))
            self.coeffs = nn.Parameter(torch.randn(self.input_size, self.output_size, self.num_knots))

    def extend_knots(self):
        """Add new knots to the spline."""
        self.num_knots += 1
        self.knots = nn.Parameter(torch.linspace(-1, 1, self.num_knots))
        self.coeffs = nn.Parameter(torch.randn(self.input_size, self.output_size, self.num_knots))

    def refine_coeffs(self):
        """Adjust coefficients for local refinement."""
        with torch.no_grad():
            self.coeffs += torch.randn_like(self.coeffs) * 0.01

    def increase_capacity(self):
        """Increase the capacity of the layer."""
        with torch.no_grad():
            self.coeffs = nn.Parameter(torch.cat([self.coeffs, torch.randn(self.input_size, self.output_size, self.num_knots)], dim=1))

    def moderate_update(self):
        """Default update routine."""
        self.refine_coeffs()


class AdaptiveKANNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers=3, temperature=0.666):
        """
        Initialize a multi-layer KAN network with adaptive layers.
        
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): List of hidden sizes for each layer.
            output_size (int): Number of output features.
            num_layers (int): Number of KAN layers.
            temperature (float): Temperature parameter for adaptive updates.
        """
        super(AdaptiveKANNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.temperature = temperature

        # Initialize KAN layers
        self.layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(AdaptiveKANLayer(in_size, hidden_size, num_knots=10, temperature=temperature))
            in_size = hidden_size
        self.output_layer = AdaptiveKANLayer(in_size, output_size, num_knots=10, temperature=temperature)

    def forward(self, x):
        """
        Forward pass through the KAN network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: Network output of shape (batch_size, output_size).
        """
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return self.output_layer(x)

    def adaptive_train_step(self, x, y, optimizer):
        """
        Single training step with adaptive updates.
        
        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            optimizer (torch.optim.Optimizer): Optimizer for updating parameters.
        """
        optimizer.zero_grad()
        output = self.forward(x)
        loss = F.mse_loss(output, y)

        # Calculate entropy and variance for adaptive updates
        entropy = torch.mean(torch.stack([layer.calculate_entropy(layer.coeffs) for layer in self.layers]))
        variance = torch.var(torch.stack([layer.calculate_entropy(layer.coeffs) for layer in self.layers]))

        # Adaptive updates
        for layer in self.layers:
            layer.adaptive_update(entropy, variance)
        
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        return loss.item()


# Example Usage
if __name__ == "__main__":
    # Define input and output sizes
    input_size = 10
    hidden_sizes = [20, 30]
    output_size = 5

    # Create the network and optimizer
    model = AdaptiveKANNetwork(input_size, hidden_sizes, output_size, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Example data
    x = torch.randn(32, input_size)
    y = torch.randn(32, output_size)

    # Training step
    for epoch in range(100):
        loss = model.adaptive_train_step(x, y, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss}")
