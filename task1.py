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
num_hidden_nodes = 32
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.ReLU(),
            nn.Linear(64, 1))
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

while num_hidden_nodes < 129:
    # Model, Loss, Optimizer
    model = DeepCorrectorMLP(num_hidden_nodes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
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
    #plt.savefig(f'Figures/task1.2/{num_hidden_nodes}-nodes-log-loss')
    plt.close()
    
    plt.plot(np.linspace(1, epochs, epochs), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Deep Neural Network Training Loss vs. Epochs')
    plt.legend()
    #plt.savefig(f'Figures/task1.2/{num_hidden_nodes}-nodes-loss')
    plt.close()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, q_target, 'r-', label='Target')
    plt.plot(t, q_real, 'b--', label='PD Only')
    plt.plot(t, q_real_corrected, 'g:', label='PD + MLP Correction')
    plt.title(f'Deep Neural Network Trajectory Tracking with and without MLP Correction - {num_hidden_nodes} Nodes')
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.legend()
    #plt.savefig(f'Figures/task1.2/{num_hidden_nodes}')
    plt.close()
    num_hidden_nodes += 32


plt.plot(np.linspace(1, epochs, epochs), train_losses_32_nodes, label='Training Loss of 32 Nodes')
plt.plot(np.linspace(1, epochs, epochs), train_losses_64_nodes, label='Training Loss of 64 Nodes')
plt.plot(np.linspace(1, epochs, epochs), train_losses_96_nodes, label='Training Loss of 96 Nodes')
plt.plot(np.linspace(1, epochs, epochs), train_losses_128_nodes, label='Training Loss of 128 Nodes')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Deep Network - Training Loss vs. Epochs')
plt.legend()
plt.savefig(f'Figures/task1.2/Training_Loss')
plt.show()

# Plot results
plt.figure(figsize=(12, 6))
plt.figure(figsize=(12, 6))

# Target
plt.plot(t, q_target, color='red', linestyle='-', label='Target')

# 32 Nodes
plt.plot(t, q_real_32, color='blue', linestyle='--', label='PD Only on 32 Nodes')
plt.plot(t, q_real_corrected_32, color='blue', linestyle=':', label='PD + MLP Correction on 32 Nodes')

# 64 Nodes
plt.plot(t, q_real_64, color='purple', linestyle='--', label='PD Only on 64 Nodes')
plt.plot(t, q_real_corrected_64, color='purple', linestyle=':', label='PD + MLP Correction on 64 Nodes')

# 96 Nodes
plt.plot(t, q_real_96, color='orange', linestyle='--', label='PD Only on 96 Nodes')
plt.plot(t, q_real_corrected_96, color='orange', linestyle=':', label='PD + MLP Correction on 96 Nodes')

# 128 Nodes
plt.plot(t, q_real_128, color='green', linestyle='--', label='PD Only on 128 Nodes')
plt.plot(t, q_real_corrected_128, color='green', linestyle=':', label='PD + MLP Correction on 128 Nodes')

# Add title, labels, and legend
plt.title('Deep Network - Trajectory Tracking with and without MLP Correction')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.legend(loc='best')

# Save and show plot
plt.savefig('Figures/task1.2/Trajectory_Tracking.png')
plt.show()

# Assume we already have the loss arrays for each configuration
# Each list contains training losses per epoch for a specific node configuration
losses_all_nodes = [
    train_losses_32_nodes,
    train_losses_64_nodes,
    train_losses_96_nodes,
    train_losses_128_nodes
]

# Convert the list of lists to a NumPy array for easier handling
losses_all_nodes = np.array(losses_all_nodes)

# Plot the heatmap of training losses
plt.figure(figsize=(10, 6))
plt.imshow(losses_all_nodes, aspect='auto', cmap='viridis')
plt.colorbar(label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Node Configuration")
plt.title("Deep Network - Training Loss Heatmap Across Node Configurations and Epochs")
plt.yticks([0, 1, 2, 3], ['32 Nodes', '64 Nodes', '96 Nodes', '128 Nodes'])
plt.savefig('Figures/task1.2/Training_Loss_Heatmap.png')
plt.show()
