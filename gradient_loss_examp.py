import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Single trainable parameter

    def forward(self, x):
        return self.weight * x  # Linear relationship: output = weight * input

# Create model, loss, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Define input and target
x = torch.tensor(2.0)  # Input
target = torch.tensor(4.0)  # Ground truth

# Forward pass
output = model(x)  # Compute prediction
loss = criterion(output, target)  # Compute loss
print(f"Initial Loss: {loss.item()}")

# Backward pass
loss.backward()  # Compute gradients
print(f"Gradient for weight: {model.weight.grad.item()}")

# Optimizer step
optimizer.step()  # Update weights
print(f"Updated weight: {model.weight.item()}")

# Check new prediction and loss
new_output = model(x)
new_loss = criterion(new_output, target)
print(f"New Loss: {new_loss.item()}")
