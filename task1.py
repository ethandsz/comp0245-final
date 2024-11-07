import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Constants
m = 1.0  # Mass (kg)
b = 10  # Friction coefficient
k_p = 50  # Proportional gain
k_d = 10   # Derivative gain
dt = 0.01  # Time step
num_samples = 1000  # Number of samples in dataset

# Generate synthetic data for trajectory tracking
t = np.linspace(0, 10, num_samples)
q_target = np.sin(t)
dot_q_target = np.cos(t)

# Initial conditions for training data generation
q = 0
dot_q = 0
X = []
Y = []

for i in range(num_samples):
    # PD control output
    tau = k_p * (q_target[i] - q) + k_d * (dot_q_target[i] - dot_q)
    # Ideal motor dynamics (variable mass for realism)
    #m_real = m * (1 + 0.1 * np.random.randn())  # Mass varies by +/-10%
    ddot_q_real = (tau - b * dot_q) / m
    
    # Calculate error
    ddot_q_ideal = (tau) / m
    ddot_q_error = ddot_q_ideal - ddot_q_real
    
    # Store data
    X.append([q, dot_q, q_target[i], dot_q_target[i]])
    Y.append(ddot_q_error)
    
    # Update state
    dot_q += ddot_q_real * dt
    q += dot_q * dt

# Convert data for PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Dataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, x):
        return self.layers(x)
class DeepCorrectorMLP(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(DeepCorrectorMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, 1)
        )

    def forward(self, x):
        return self.layers(x)

learning_rates = [1, 1e-1, 1e-2, 1e-3, 1e-4]
num_hidden_nodes = 32
train_losses_32_nodes = []
train_losses_64_nodes = []
train_losses_96_nodes = []
train_losses_128_nodes = []

q_real_32 = []
q_real_64 = []
q_real_96 = []
q_real_128 = []

q_real_corrected_32 = []
q_real_corrected_64 = []
q_real_corrected_96 = []
q_real_corrected_128 = []
epochs = 1000

for lr in learning_rates:

    while num_hidden_nodes < 129:
        # Model, Loss, Optimizer
        model = DeepCorrectorMLP(num_hidden_nodes = 128)
        # model = MLP()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training Loop
        train_losses = []
        
        
        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
            epoch_loss += loss.item()
        
            train_losses.append(epoch_loss / len(train_loader))
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.6f}')
        
        # Testing Phase: Simulate trajectory tracking
        q_test = 0
        dot_q_test = 0
        q_real = []
        q_real_corrected = []
        
        
        # integration with only PD Control
        for i in range(len(t)):
            tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
            ddot_q_real = (tau - b * dot_q_test) / m
            dot_q_test += ddot_q_real * dt
            q_test += dot_q_test * dt
            q_real.append(q_test)
        
        q_test = 0
        dot_q_test = 0
        for i in range(len(t)):
            # Apply MLP correction
            tau = k_p * (q_target[i] - q_test) + k_d * (dot_q_target[i] - dot_q_test)
            inputs = torch.tensor([q_test, dot_q_test, q_target[i], dot_q_target[i]], dtype=torch.float32)
            correction = model(inputs.unsqueeze(0)).item()
            ddot_q_corrected =(tau - b * dot_q_test + correction) / m
            dot_q_test += ddot_q_corrected * dt
            q_test += dot_q_test * dt
            q_real_corrected.append(q_test)

            if(num_hidden_nodes == 32):
                train_losses_32_nodes = train_losses
                q_real_32 = q_real
                q_real_corrected_32 = q_real_corrected
            elif(num_hidden_nodes == 64):
                train_losses_64_nodes = train_losses
                q_real_64 = q_real
                q_real_corrected_64 = q_real_corrected
            elif(num_hidden_nodes == 96):
                train_losses_96_nodes = train_losses
                q_real_96 = q_real
                q_real_corrected_96 = q_real_corrected
            elif(num_hidden_nodes == 128):
                train_losses_128_nodes = train_losses
                q_real_128 = q_real
                q_real_corrected_128 = q_real_corrected
                
        plt.plot(np.linspace(1, epochs, epochs), np.log(train_losses), label='Log Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log(Loss)')
        plt.title('Deep Neural Network Logarithmic Training Loss vs. Epochs')
        plt.legend()
        plt.savefig(f'Figures/task1.3/Deep-network-128-nodes-log-loss-lr={lr}.png')
        plt.close()
        
        plt.plot(np.linspace(1, epochs, epochs), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Deep Neural Network Training Loss vs. Epochs')
        plt.legend()
        plt.savefig(f'Figures/task1.3/Deep-network-128-nodes-loss-lr={lr}.png')
        plt.close()
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(t, q_target, 'r-', label='Target')
        plt.plot(t, q_real, 'b--', label='PD Only')
        plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
        plt.title(f'Deep Neural Network Trajectory Tracking with and without MLP Correction - 128 Nodes - {lr} Learning rate')
        plt.xlabel('Time [s]')
        plt.ylabel('Position')
        plt.legend()
        plt.savefig(f'Figures/task1.3/Deep-network-128-nodes-lr={lr}.png')
        plt.close()