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
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import pandas as pd



# Set the model type: "neural_network" or "random_forest"
neural_network_or_random_forest = "random_forest"  # Change to "random_forest" to use Random Forest models
depth = 10

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

def plot_velocities(time, velocities, original, method, test_num):
    num_joints = velocities.shape[1]  # Number of joints

    # Loop over each joint and create a separate plot
    for joint_idx in range(num_joints):
        # Set up a 1x2 grid for smoothed and original plots for the current joint
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Joint {joint_idx + 1} Velocities Over Time", fontsize=16)

        # Smoothed velocity plot
        axes[0].plot(time, velocities[:, joint_idx], label='Smoothed', color='blue')
        axes[0].set_title(f"Joint {joint_idx + 1} - Smoothed with {method}")
        axes[0].set_ylabel("Velocity")
        axes[0].set_xlabel("Time")
        axes[0].legend()

        # Original velocity plot
        axes[1].plot(time, original[:, joint_idx], label='Original', color='orange')
        axes[1].set_title(f"Joint {joint_idx + 1} - Original")
        axes[1].set_xlabel("Time")
        axes[1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{method}/Test {test_num}/Joint {joint_idx + 1}/velocity_comparison"
        try:
            plt.savefig(save_path, dpi=300)
        except:
            print("Directory doesnt exist, most likely need to add extra test folders.")

    plt.close()

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
    number_of_goal_positions_to_test = 5
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
    smoothing_tests = ["Gaussian Filter", "EMA","original"]
    end_position_losses = pd.DataFrame(columns=smoothing_tests)

    for goal_position in goal_positions:
        losses = []
        for smoothing_test in smoothing_tests:
            print("Testing new goal position------------------------------------")
            print(f"Goal position: {goal_position}")
            test_num = goal_positions.index(goal_position) + 1
    
            # Initialize the simulation
            sim.ResetPose()
            sim.pybullet_client.resetBasePositionAndOrientation(1, goal_position, sim.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
    
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
            original_qd_des = qd_des_over_time_clipped
            # Parameters
            if smoothing_test == "EMA":
                alpha = 2 / (1000 + 1)  
                
                # Initialize an empty array to hold the smoothed joint positions
                predicted_joint_positions_ema = np.zeros_like(predicted_joint_positions_over_time)
                
                # Calculate EMA for each joint
                for joint_idx in range(7):
                    ema = np.zeros(predicted_joint_positions_over_time.shape[0])
                    ema[0] = predicted_joint_positions_over_time[0, joint_idx]  # Initial EMA value
                
                    # Compute EMA for each time step
                    for t in range(1, predicted_joint_positions_over_time.shape[0]):
                        ema[t] = alpha * predicted_joint_positions_over_time[t, joint_idx] + (1 - alpha) * ema[t-1]
                
                    plt.plot(test_time_array[:], ema, label="EMA")
                    plt.plot(test_time_array, predicted_joint_positions_over_time[:,joint_idx], label="Original Position")
                    plt.title(f"Joint {joint_idx + 1}: Original vs. Smoothed ({smoothing_test})")
                    plt.legend()
                    save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{smoothing_test}/Test {test_num}/Joint {joint_idx + 1}/position_comparison"
                    try:
                        plt.savefig(save_path, dpi=300)
                    except:
                        print("Directory doesnt exist, most likely need to add extra test folders.")
                    plt.close()
                
                    predicted_joint_positions_ema[:, joint_idx] = ema
                    
                predicted_joint_positions_over_time = predicted_joint_positions_ema
                
                # Compute qd_des_over_time by numerically differentiating the EMA-smoothed joint positions
                qd_des_over_time = np.gradient(predicted_joint_positions_ema, axis=0, edge_order=2) / time_step
                
                # Clip the joint velocities to the joint limits
                qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))
            
            elif smoothing_test == "Gaussian Filter":
                predicted_joint_positions_gaussian = np.zeros_like(predicted_joint_positions_over_time)
                for joint_idx in range(7):
                    y_gaussian = gaussian_filter1d(predicted_joint_positions_over_time[:,joint_idx], sigma=600)
                    plt.plot(test_time_array, predicted_joint_positions_over_time[:,joint_idx], label="Original Position")
                    plt.plot(test_time_array, y_gaussian, label="Gaussian Filter", color='green')
                    plt.legend()
                    plt.title(f"Joint {joint_idx + 1}: Original vs. Smoothed ({smoothing_test})")
                    save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{smoothing_test}/Test {test_num}/Joint {joint_idx + 1}/position_comparison"
                    try:
                        plt.savefig(save_path, dpi=300)
                    except:
                        print("Directory doesnt exist, most likely need to add extra test folders.")
                    plt.close()
                    predicted_joint_positions_gaussian[:,joint_idx] = y_gaussian
                
                predicted_joint_positions_over_time = predicted_joint_positions_gaussian

                qd_des_over_time = np.gradient(predicted_joint_positions_gaussian, axis=0, edge_order=2) / time_step
                # Clip the joint velocities to the joint limits
                qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))
                plot_velocities(test_time_array, qd_des_over_time_clipped, original_qd_des, smoothing_test, test_num)


            q_mes_all = []
            q_des_all = []
    
            qd_mes_all = []
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
                qd_mes_all.append(qd_mes)
    
            # After the trajectory, compute the final cartesian position
            final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
            final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
            print(f"Final computed cartesian position: {final_cartesian_pos}")
            # Compute position error
            position_error = np.linalg.norm(final_cartesian_pos - goal_position)
            print(f"Position error between computed position and goal: {position_error}")
            losses.append(position_error)
            q_mes_all = np.array(q_mes_all)
            q_des_all = np.array(q_des_all)
            qd_mes_all = np.array(qd_mes_all)
            qd_des_over_time = np.array(qd_des_over_time)
            qd_des_over_time_clipped = np.array(qd_des_over_time_clipped)
    
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
    
                fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column
    
                # 1. Plot Measured vs Desired Positions in the first subplot
                axs[0].plot(test_time_array[3:-1], q_mes_all[3:, joint_idx], label="Measured Position", color="blue", linestyle='-', linewidth=2)
                axs[0].plot(test_time_array[3:-1], q_des_all[3:, joint_idx], label="Desired Position", color="red", linestyle='--', linewidth=2)
                axs[0].set_title(f"Comparison of Measured and Desired Positions Over Time for Joint {joint_idx + 1}", fontsize=16)
                axs[0].set_xlabel("Time (s)", fontsize=14)
                axs[0].set_ylabel("Position (m)", fontsize=14)
                axs[0].grid(True, linestyle="--", alpha=0.7)
                axs[0].legend(fontsize=12)
    
                # 2. Plot Squared Loss in the second subplot
                squared_loss = (q_des_all[3:, joint_idx] - q_mes_all[3:, joint_idx]) ** 2
                axs[1].plot(test_time_array[3:-1], squared_loss, label="Squared Loss", color="purple", linestyle='--', linewidth=2)
                axs[1].set_title("Squared Loss Between Desired and Measured Positions Over Time", fontsize=16)
                axs[1].set_xlabel("Time (s)", fontsize=14)
                axs[1].set_ylabel("Loss", fontsize=14)
                axs[1].grid(True, linestyle="--", alpha=0.7)
                axs[1].legend(fontsize=12)
    
                # Adjust layout to avoid overlap
                plt.tight_layout()
                save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{smoothing_test}/Test {test_num}/Joint {joint_idx + 1}/position_loss_des_mes"
                try:
                    plt.savefig(save_path, dpi=300)
                except:
                    print("Directory doesnt exist, most likely need to add extra test folders.")
                plt.close()
    
    
                fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column
    
                # 1. Plot Measured vs Desired Positions in the first subplot
                axs[0].plot(test_time_array[3:-1], qd_mes_all[3:, joint_idx], label="Measured Velocity", color="blue", linestyle='-', linewidth=2)
                axs[0].plot(test_time_array[3:-1], qd_des_over_time[3:-1, joint_idx], label="Desired Velocity", color="green", linestyle='--', linewidth=2)
                axs[0].set_title(f"Comparison of Measured and Desired Velocity Over Time for Joint {joint_idx + 1}", fontsize=16)
                axs[0].set_xlabel("Time (s)", fontsize=14)
                axs[0].set_ylabel("Velocity (m/s)", fontsize=14)
                axs[0].grid(True, linestyle="--", alpha=0.7)
                axs[0].legend(fontsize=12)
    
                # 2. Plot Squared Loss in the second subplot
                squared_loss = (qd_des_over_time[3:-1, joint_idx] - qd_mes_all[3:, joint_idx]) ** 2
                axs[1].plot(test_time_array[3:-1], squared_loss, label="Squared Loss", color="purple", linestyle='--', linewidth=2)
                axs[1].set_title("Squared Loss Between Desired and Measured Velocities Over Time", fontsize=16)
                axs[1].set_xlabel("Time (s)", fontsize=14)
                axs[1].set_ylabel("Loss", fontsize=14)
                axs[1].grid(True, linestyle="--", alpha=0.7)
                axs[1].legend(fontsize=12)
    
                # Adjust layout to avoid overlap
                plt.tight_layout()
                save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{smoothing_test}/Test {test_num}/Joint {joint_idx + 1}/velocity_loss_des_mes"
                try:
                    plt.savefig(save_path, dpi=300)
                except:
                    print("Directory doesnt exist, most likely need to add extra test folders.")
                plt.close()
    
    
            # Set the camera position a little further away to capture the full area
            camera_distance = 1.0  # Adjust distance based on field of view and bounds
            camera_eye_position = [goal_position[0], goal_position[1] - camera_distance, goal_position[2] + 0.5]
            camera_target_position = goal_position  # Focus on the center of the bounds
    
    
    
            view_matrix = sim.pybullet_client.computeViewMatrix(
                cameraEyePosition=camera_eye_position,
                cameraTargetPosition=camera_target_position,
                cameraUpVector=[0, 0, 1]
            )
            # Define camera parameters
            width, height = 640, 480
            fov = 60  # Field of view
            aspect = width / height
            near = 0.1  # Near clipping plane
            far = 10.0  # Far clipping plane
            projection_matrix = sim.pybullet_client.computeProjectionMatrixFOV(
                fov=fov,
                aspect=aspect,
                nearVal=near,
                farVal=far
            )
    
            # Capture the image
            image = sim.pybullet_client.getCameraImage(width, height, view_matrix, projection_matrix)
            rgb_array = np.array(image[2])[:, :, :3]  # Extract the RGB data
    
            # Convert to an image and save
            screenshot = Image.fromarray(rgb_array)
            # screenshot.save(f"Figures/task3.3/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/end_position_screenshot.png")
    
    
    
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(desired_cartesian_positions_over_time[3:, 0], desired_cartesian_positions_over_time[3:, 1], desired_cartesian_positions_over_time[3:, 2], label='Desired Trajectory')
            ax.plot(measured_cartesian_positions_over_time[3:, 0], measured_cartesian_positions_over_time[3:, 1], measured_cartesian_positions_over_time[3:, 2], label="Measured Trajectory")
            ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label=f'Goal: ({goal_position[0]:.3f}, {goal_position[1]:.3f}, {goal_position[2]:.3f})')
            ax.scatter(measured_cartesian_positions_over_time[-1, 0], measured_cartesian_positions_over_time[-1, 1], measured_cartesian_positions_over_time[-1, 2], color='green', label=f"End:  ({measured_cartesian_positions_over_time[-1, 0]:.3f}, {measured_cartesian_positions_over_time[-1, 1]:.3f}, {measured_cartesian_positions_over_time[-1, 2]:.3f})")
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            ax.set_title('Predicted Cartesian Trajectory')
            plt.legend(loc=9)
            save_path = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/{smoothing_test}/Test {test_num}/Cartesian_trajectory"
            try:
                plt.savefig(save_path, dpi=300)
            except:
                print("Directory doesnt exist, most likely need to add extra test folders.")
            plt.close()
        end_position_losses.loc[len(end_position_losses)] = losses
    lossSavePath = f"Figures/task3.3/{'Depth 2' if depth == 2 else 'Depth 10'}/data.json"

    end_position_losses.to_json(lossSavePath, orient="records", lines=True)



if __name__ == '__main__':
    main()