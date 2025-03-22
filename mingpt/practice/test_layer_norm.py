import torch
import numpy as np
import unittest
from mingpt.practice.manual_backprop import CustomLayerNormFunction

class TestCustomLayerNorm(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Default epsilon for numerical stability
        self.eps = 1e-5
    
    def test_forward_2d(self, custom_ln=CustomLayerNormFunction):
        # Test with 2D tensor (batch_size, features)
        batch_size, features = 4, 8
        x = torch.randn(batch_size, features, requires_grad=True)
        gamma = torch.ones(features, requires_grad=True)
        beta = torch.zeros(features, requires_grad=True)
        
        # Our custom implementation
        custom_output = custom_ln.apply(x, gamma, beta, self.eps)
        print(custom_output)
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma.clone()
        torch_ln.bias.data = beta.clone()
        torch_output = torch_ln(x)
        
        # Check that outputs match
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5))
        
    def test_forward_3d(self):
        # Test with 3D tensor (batch_size, seq_len, features)
        batch_size, seq_len, features = 2, 5, 10
        x = torch.randn(batch_size, seq_len, features, requires_grad=True)
        gamma = torch.ones(features, requires_grad=True)
        beta = torch.zeros(features, requires_grad=True)
        
        # Our custom implementation
        custom_output = CustomLayerNormFunction.apply(x, gamma, beta, self.eps)
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma.clone()
        torch_ln.bias.data = beta.clone()
        torch_output = torch_ln(x)
        
        # Check that outputs match
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5))
        
    def test_forward_with_non_default_params(self):
        # Test with non-default gamma and beta values
        batch_size, features = 3, 6
        x = torch.randn(batch_size, features, requires_grad=True)
        gamma = torch.randn(features, requires_grad=True)  # Random gamma
        beta = torch.randn(features, requires_grad=True)   # Random beta
        
        # Our custom implementation
        custom_output = CustomLayerNormFunction.apply(x, gamma, beta, self.eps)
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma.clone()
        torch_ln.bias.data = beta.clone()
        torch_output = torch_ln(x)
        
        # Check that outputs match
        self.assertTrue(torch.allclose(custom_output, torch_output, rtol=1e-5, atol=1e-5))
    
    def test_backward_2d(self, custom_ln=CustomLayerNormFunction):
        # Test gradients with 2D tensor
        batch_size, features = 4, 8
        x = torch.randn(batch_size, features, requires_grad=True)
        gamma = torch.ones(features, requires_grad=True)
        beta = torch.zeros(features, requires_grad=True)
        
        # Clone parameters for PyTorch implementation
        x_torch = x.clone().detach().requires_grad_(True)
        gamma_torch = gamma.clone().detach().requires_grad_(True)
        beta_torch = beta.clone().detach().requires_grad_(True)
        
        # Forward pass with custom implementation
        custom_output = custom_ln.apply(x, gamma, beta, self.eps)
        # Compute loss and backward
        custom_loss = custom_output.sum()
        custom_loss.backward()
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma_torch
        torch_ln.bias.data = beta_torch
        torch_output = torch_ln(x_torch)
        # Compute loss and backward
        torch_loss = torch_output.sum()
        torch_loss.backward()

        # Check gradients
        self.assertTrue(torch.allclose(x.grad, x_torch.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(gamma.grad, torch_ln.weight.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(beta.grad, torch_ln.bias.grad, rtol=1e-4, atol=1e-4))
        
    def test_backward_3d(self):
        # Test gradients with 3D tensor
        batch_size, seq_len, features = 2, 5, 10
        x = torch.randn(batch_size, seq_len, features, requires_grad=True)
        gamma = torch.ones(features, requires_grad=True)
        beta = torch.zeros(features, requires_grad=True)
        
        # Clone parameters for PyTorch implementation
        x_torch = x.clone().detach().requires_grad_(True)
        gamma_torch = gamma.clone().detach().requires_grad_(True)
        beta_torch = beta.clone().detach().requires_grad_(True)
        
        # Forward pass with custom implementation
        custom_output = CustomLayerNormFunction.apply(x, gamma, beta, self.eps)
        # Create a more complex loss to test gradient flow
        custom_loss = (custom_output ** 2).mean()
        custom_loss.backward()
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma_torch
        torch_ln.bias.data = beta_torch
        torch_output = torch_ln(x_torch)
        # Create the same loss
        torch_loss = (torch_output ** 2).mean()
        torch_loss.backward()
        
        # Check gradients
        self.assertTrue(torch.allclose(x.grad, x_torch.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(gamma.grad, torch_ln.weight.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(beta.grad, torch_ln.bias.grad, rtol=1e-4, atol=1e-4))
        
    def test_backward_with_non_default_params(self):
        # Test gradients with non-default gamma and beta
        batch_size, features = 3, 6
        x = torch.randn(batch_size, features, requires_grad=True)
        gamma = torch.randn(features, requires_grad=True)  # Random gamma
        beta = torch.randn(features, requires_grad=True)   # Random beta
        
        # Clone parameters for PyTorch implementation
        x_torch = x.clone().detach().requires_grad_(True)
        gamma_torch = gamma.clone().detach().requires_grad_(True)
        beta_torch = beta.clone().detach().requires_grad_(True)
        
        # Forward pass with custom implementation
        custom_output = CustomLayerNormFunction.apply(x, gamma, beta, self.eps)
        # Create a complex loss
        custom_loss = (custom_output * torch.randn_like(custom_output)).sum()
        custom_loss.backward()
        
        # PyTorch's implementation
        torch_ln = torch.nn.LayerNorm(features, elementwise_affine=True, eps=self.eps)
        torch_ln.weight.data = gamma_torch
        torch_ln.bias.data = beta_torch
        torch_output = torch_ln(x_torch)
        # Create the same loss with same random values
        torch.manual_seed(42)  # Reset seed to get same random values
        torch_loss = (torch_output * torch.randn_like(torch_output)).sum()
        torch_loss.backward()
        
        # Check gradients
        self.assertTrue(torch.allclose(x.grad, x_torch.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(gamma.grad, torch_ln.weight.grad, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(beta.grad, torch_ln.bias.grad, rtol=1e-4, atol=1e-4))

if __name__ == '__main__':
    unittest.main() 