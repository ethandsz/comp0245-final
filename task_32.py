import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin
import threading
import pickle
import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestRegressor
import joblib  # For saving and loading models

# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "random_forest" to use Random Forest models

# MLP Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),  # Input layer to hidden layer (4 inputs: time + goal positions)
            nn.ReLU(),
            nn.Linear(128, 1)   # Hidden layer to output layer
        )

    def forward(self, x):
        return self.model(x)

def main():
    # Load the saved data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, 'data.pkl')  # Replace with your actual filename
    if not os.path.isfile(filename):
        print(f"Error: File {filename} not found in {script_dir}")
        return
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        time_array = np.array(data['time'])            # Shape: (N,)
        # Optional: Normalize time data for better performance
        # time_array = (time_array - time_array.min()) / (time_array.max() - time_array.min())

    # Load all the models in a list
    models = []
    if neural_network_or_random_forest == "neural_network":
        for joint_idx in range(7):
            # Instantiate the model
            model = MLP()
            # Load the saved model
            model_filename = os.path.join(script_dir, f'neuralq{joint_idx+1}.pt')
            model.load_state_dict(torch.load(model_filename))
            model.eval()
            models.append(model)
    elif neural_network_or_random_forest == "random_forest":
        for joint_idx in range(7):
            # Load the saved Random Forest model
            model_filename = os.path.join(script_dir, f'rf_joint{joint_idx+1}.joblib')
            model = joblib.load(model_filename)
            models.append(model)
    else:
        print("Invalid model type specified. Please set neural_network_or_random_forest to 'neural_network' or 'random_forest'")
        return

    # Generate a new goal position
    goal_position_bounds = {
        'x': (0.6, 0.8),
        'y': (-0.1, 0.1),
        'z': (0.12, 0.12)
    }
    # Create a set of goal positions
    number_of_goal_positions_to_test = 1
    goal_positions = []
    for i in range(number_of_goal_positions_to_test):
        goal_positions.append([
            np.random.uniform(*goal_position_bounds['x']),
            np.random.uniform(*goal_position_bounds['y']),
            np.random.uniform(*goal_position_bounds['z'])
        ])

    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration for the simulation
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=root_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, root_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    controlled_frame_name = "panda_link8"
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    print(f"Initial joint angles: {init_joint_angles}")

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # PD controller gains
    kp = 1000  # Proportional gain
    kd = 100   # Derivative gain

    # Get joint velocity limits
    joint_vel_limits = sim.GetBotJointsVelLimit()

    time_step = sim.GetTimeStep()
    # Generate test time array
    test_time_array = np.arange(time_array.min(), time_array.max(), time_step)

    for goal_position in goal_positions:
        print("Testing new goal position------------------------------------")
        print(f"Goal position: {goal_position}")

        # Initialize the simulation
        sim.ResetPose()
        current_time = 0  # Initialize current time

        # Create test input features
        test_goal_positions = np.tile(goal_position, (len(test_time_array), 1))  # Shape: (num_points, 3)
        test_input = np.hstack((test_time_array.reshape(-1, 1), test_goal_positions))  # Shape: (num_points, 4)

        # Predict joint positions for the new goal position
        predicted_joint_positions_over_time = np.zeros((len(test_time_array), 7))  # Shape: (num_points, 7)

        for joint_idx in range(7):
            if neural_network_or_random_forest == "neural_network":
                # Prepare the test input
                test_input_tensor = torch.from_numpy(test_input).float()  # Shape: (num_points, 4)

                # Predict joint positions using the neural network
                with torch.no_grad():
                    predictions = models[joint_idx](test_input_tensor).numpy().flatten()  # Shape: (num_points,)
            elif neural_network_or_random_forest == "random_forest":
                # Predict joint positions using the Random Forest
                predictions = models[joint_idx].predict(test_input)  # Shape: (num_points,)

            # Store the predicted joint positions
            predicted_joint_positions_over_time[:, joint_idx] = predictions

        # Compute qd_des_over_time by numerically differentiating the predicted joint positions
        qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
        # Clip the joint velocities to the joint limits
        qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))
        
        q_mes_all = []
        q_des_all = []
        # Data collection loop
        while current_time < test_time_array.max():
            # Measure current state
            q_mes = sim.GetMotorAngles(0)
            qd_mes = sim.GetMotorVelocities(0)
            qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
            # Get the index corresponding to the current time
            current_index = int(current_time / time_step)
            if current_index >= len(test_time_array):
                current_index = len(test_time_array) - 1

            # Get q_des and qd_des_clip from predicted data
            q_des = predicted_joint_positions_over_time[current_index, :]  # Shape: (7,)
            qd_des_clip = qd_des_over_time_clipped[current_index, :]      # Shape: (7,)

            # Control command
            tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des_clip, kp, kd)
            cmd.SetControlCmd(tau_cmd, ["torque"] * 7)  # Set the torque command
            sim.Step(cmd, "torque")  # Simulation step with torque command

            # Keyboard event handling
            keys = sim.GetPyBulletClient().getKeyboardEvents()
            qKey = ord('q')

            # Exit logic with 'q' key
            if qKey in keys and keys[qKey] & sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("Exiting simulation.")
                break

            # Time management
            time.sleep(time_step)  # Control loop timing
            current_time += time_step
            
            q_mes_all.append(q_mes)
            q_des_all.append(q_des)

        # After the trajectory, compute the final cartesian position
        final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
        final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
        print(f"Final computed cartesian position: {final_cartesian_pos}")
        # Compute position error
        position_error = np.linalg.norm(final_cartesian_pos - goal_position)
        print(f"Position error between computed position and goal: {position_error}")
        
        q_mes_all = np.array(q_mes_all)
        q_des_all = np.array(q_des_all)

        desired_cartesian_positions_over_time = []
        measured_cartesian_positions_over_time = []
        
        for i in range(len(test_time_array)- 1):
            joint_positions = q_des_all[i, :]
            cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
            desired_cartesian_positions_over_time.append(cartesian_pos.copy())
            
        for i in range(len(test_time_array) - 1):
            joint_positions = q_mes_all[i, :]
            cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
            measured_cartesian_positions_over_time.append(cartesian_pos.copy())
        
        desired_cartesian_positions_over_time = np.array(desired_cartesian_positions_over_time)
        measured_cartesian_positions_over_time = np.array(measured_cartesian_positions_over_time)

        for joint_idx in range(7):
            plt.figure(figsize=(10, 6))
            
            # Plot measured and desired values with labels and styles
            plt.plot(test_time_array[3:-1], q_mes_all[3:, joint_idx], label="Measured Position", color="blue", linestyle='-', linewidth=2)
            plt.plot(test_time_array[3:-1], q_des_all[3:, joint_idx], label="Desired Position", color="red", linestyle='--', linewidth=2)
            # Title and labels
            plt.title(f"Comparison of Measured and Desired Positions Over Time For Joint {joint_idx + 1}", fontsize=16)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel("Position", fontsize=14)
            
            # Grid, legend, and display
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()  # Adjust layout to prevent clipping
            
            plt.show()
            

            squared_loss = (q_des_all[3:, joint_idx] - q_mes_all[3:, joint_idx]) ** 2
            plt.plot(test_time_array[3:-1], squared_loss, label="Squared Loss", color="purple", linestyle='--', linewidth=2)

            # Title and labels
            plt.title("Loss Between Desired and Measured Positions Over Time", fontsize=16)
            plt.xlabel("Time (s)", fontsize=14)
            plt.ylabel("Loss", fontsize=14)
            
            # Grid, legend, and display
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()  # Adjust layout to prevent clipping
            
            plt.show()



        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(desired_cartesian_positions_over_time[:, 0], desired_cartesian_positions_over_time[:, 1], desired_cartesian_positions_over_time[:, 2], label='Desired Trajectory')
        ax.plot(measured_cartesian_positions_over_time[:, 0], measured_cartesian_positions_over_time[:, 1], measured_cartesian_positions_over_time[:, 2], label='Measured Trajectory for Random Forest')
        ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label='Goal Position')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Predicted Cartesian Trajectory')
        plt.legend()
        save_path = f"Figures/task3.2"
        # try:
        #     plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # except:
        #     print("Directory doesnt exist, most likely need to add extra test folders.")
        plt.show()

if __name__ == '__main__':
    main()