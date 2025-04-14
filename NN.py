import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import Dataset as df

X_train = df.getTrainSetX()
y_train = df.getTrainSetY()

X_test = df.getTestSetX()
y_test = df.getTestSetY()


# Convert preprocessed data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize the model
input_size = X_train_tensor.shape[1]
hidden_size = 32  # You can adjust this
output_size = 1
model = NeuralNet(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            labels = labels.view(-1, 1)  # Reshape for binary classification
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.view(-1, 1)  # Reshape for binary classification
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()  # Apply threshold
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())
    y_true = torch.tensor(y_true).flatten()
    y_pred = torch.tensor(y_pred).flatten()
    accuracy = (y_true == y_pred).sum().item() / len(y_true)
    print(f"Accuracy: {accuracy:.4f}")

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
evaluate_model(model, test_loader)
