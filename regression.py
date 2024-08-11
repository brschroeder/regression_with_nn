import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate synthetic non-linear data
def generate_data(num_points=100):
    X = np.linspace(-2, 2, num_points)
    y = np.sin(3 * X) + np.random.normal(0, 0.1, num_points)
    return X, y

# Prepare the data
X, y = generate_data()
X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape (num_points, 1)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (num_points, 1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(1, 10)  # Hidden layer with 10 neurons
        self.output = nn.Linear(10, 1)  # Output layer
        
    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Apply ReLU activation function
        x = self.output(x)  # Linear output
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(X_train).detach().numpy()

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Original Data')
plt.plot(X, predicted, color='red', label='Fitted Line')
plt.legend()
plt.show()
