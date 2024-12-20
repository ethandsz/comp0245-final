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
#neural_network_or_random_forest = "neural_network" #"neural_network"  # Change to "random_forest" to use Random Forest models

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

    predicted_cartesian_for_nn = []
    predicted_cartesian_for_rf = []

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

    for modelName in ['random_forest','neural_network']:
        neural_network_or_random_forest = modelName
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

            for joint_idx in range(7):
                plt.figure(figsize=(10, 6))

                # Plot settings
                plt.plot(test_time_array[5:], predicted_joint_positions_over_time[5:, joint_idx], color="blue", linewidth=2)
                plt.title(f"Predicted Joint Position for Joint {joint_idx+1}", fontsize=16)
                plt.xlabel("Time (s)", fontsize=14)
                plt.ylabel("Position (m)", fontsize=14)
                plt.grid(True, linestyle="--", alpha=0.7)

                # Save each plot as an image file
                save_path = f"Figures/task3.1/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/Predictions/Joint {joint_idx+1}/position_over_time.png"
                try:
                    plt.savefig(save_path, dpi=300)
                except:
                    print("Directory doesnt exist, most likely need to add extra test folders.")
                plt.close()


                plt.figure(figsize=(10, 6))

                # Plot settings
                plt.plot(test_time_array[5:], np.gradient(predicted_joint_positions_over_time[5:, joint_idx], test_time_array[5:]), color="blue", linewidth=2)
                plt.title(f"Predicted Joint Velocity for Joint {joint_idx+1}", fontsize=16)
                plt.xlabel("Time (s)", fontsize=14)
                plt.ylabel("Velocity (m/s)", fontsize=14)
                plt.grid(True, linestyle="--", alpha=0.7)

                # Save each plot as an image file
                save_path = f"Figures/task3.1/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/Predictions/Joint {joint_idx+1}/velocity_over_time.png"
                try:
                    plt.savefig(save_path, dpi=300)
                except:
                    print("Directory doesnt exist, most likely need to add extra test folders.")
                plt.close()

            final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)

            # Compute forward kinematics
            final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)

            print(f"Goal position: {goal_position}")
            print(f"Computed cartesian position: {final_cartesian_pos}")
            print(f"Predicted joint positions at final time step: {final_predicted_joint_positions}")

            # Compute position error
            position_error = np.linalg.norm(final_cartesian_pos - goal_position)
            print(f"Position error between computed position and goal: {position_error}")

            # Optional: Visualize the cartesian trajectory over time
            cartesian_positions_over_time = []
            for i in range(len(test_time_array)):
                joint_positions = predicted_joint_positions_over_time[i, :]
                cartesian_pos, _ = dyn_model.ComputeFK(joint_positions, controlled_frame_name)
                cartesian_positions_over_time.append(cartesian_pos.copy())

            cartesian_positions_over_time = np.array(cartesian_positions_over_time)  # Shape: (num_points, 3)

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(test_time_array[5:], cartesian_positions_over_time[5:, 0], label='X Position')
            plt.plot(test_time_array[5:], cartesian_positions_over_time[5:, 1], label='Y Position')
            plt.plot(test_time_array[5:], cartesian_positions_over_time[5:, 2], label='Z Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Cartesian Position (m)')
            plt.title('Predicted Cartesian Positions Over Time')
            plt.legend()
            plt.grid(True)
            save_path = f"Figures/task3.1/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/Predictions/Cartesian Positions/2D-graph.png"
            try:
                plt.savefig(save_path, dpi=300)
            except:
                print("Directory doesnt exist, most likely need to add extra test folders.")
            plt.close()

            # Plot x, y, z positions over time
            plt.figure(figsize=(10, 5))
            plt.plot(test_time_array[5:], np.gradient(cartesian_positions_over_time[5:, 0], test_time_array[5:]), label='X Position')
            plt.plot(test_time_array[5:], np.gradient(cartesian_positions_over_time[5:, 1], test_time_array[5:]), label='Y Position')
            plt.plot(test_time_array[5:], np.gradient(cartesian_positions_over_time[5:, 2], test_time_array[5:]), label='Z Position')
            plt.xlabel('Time (s)')
            plt.ylabel('Cartesian Velocity (m/s)')
            plt.title('Predicted Cartesian Velocity Over Time')
            plt.legend()
            plt.grid(True)
            save_path = f"Figures/task3.1/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/Predictions/Cartesian Positions/2D-graph-velocities.png"
            try:
                plt.savefig(save_path, dpi=300)
            except:
                print("Directory doesnt exist, most likely need to add extra test folders.")
            plt.close()

            # Plot the trajectory in 3D space
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(cartesian_positions_over_time[5:, 0], cartesian_positions_over_time[5:, 1], cartesian_positions_over_time[5:, 2], label='Predicted Trajectory')
            ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label=f'Goal: ({goal_position[0]:.3f}, {goal_position[1]:.3f}, {goal_position[2]:.3f})')
            ax.scatter(cartesian_positions_over_time[-1, 0], cartesian_positions_over_time[-1, 1], cartesian_positions_over_time[-1, 2], color='green', label=f"End: ({cartesian_positions_over_time[-1, 0]:.3f}, {cartesian_positions_over_time[-1, 1]:.3f}, {cartesian_positions_over_time[-1, 2]:.3f})")
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            ax.set_title('Predicted Cartesian Trajectory')
            plt.legend(loc=9)
            save_path = f"Figures/task3.1/{'Neural Network' if neural_network_or_random_forest == 'neural_network' else 'Random Forest'}/Test {goal_positions.index(goal_position) + 1}/Predictions/Cartesian Positions/3D-graph.png"
            try:
                plt.savefig(save_path, dpi=300)
            except:
                print("Directory doesnt exist, most likely need to add extra test folders.")
            plt.close()

            if(neural_network_or_random_forest == 'neural_network'):
                predicted_cartesian_for_nn.append(cartesian_positions_over_time)
            else:
                predicted_cartesian_for_rf.append(cartesian_positions_over_time)


            # Compute qd_des_over_time by numerically differentiating the predicted joint positions
            qd_des_over_time = np.gradient(predicted_joint_positions_over_time, axis=0, edge_order=2) / time_step
            # Clip the joint velocities to the joint limits
            qd_des_over_time_clipped = np.clip(qd_des_over_time, -np.array(joint_vel_limits), np.array(joint_vel_limits))

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

            # After the trajectory, compute the final cartesian position
            final_predicted_joint_positions = predicted_joint_positions_over_time[-1, :]  # Shape: (7,)
            final_cartesian_pos, final_R = dyn_model.ComputeFK(final_predicted_joint_positions, controlled_frame_name)
            print(f"Final computed cartesian position: {final_cartesian_pos}")
            # Compute position error
            position_error = np.linalg.norm(final_cartesian_pos - goal_position)
            print(f"Position error between computed position and goal: {position_error}")
        del sim
        del dyn_model


    for goal_position in goal_positions:
        # Plot x, y, z positions over time
        i = goal_positions.index(goal_position)
        plt.figure(figsize=(10, 5))
        plt.plot(test_time_array[5:], predicted_cartesian_for_nn[goal_positions.index(goal_position)][5:, 0], label='NN X Position')
        plt.plot(test_time_array[5:], predicted_cartesian_for_nn[goal_positions.index(goal_position)][5:, 1], label='NN Y Position')
        plt.plot(test_time_array[5:], predicted_cartesian_for_nn[i][5:, 2], label='NN Z Position')
        plt.plot(test_time_array[5:], predicted_cartesian_for_rf[i][5:, 0], label='RF X Position')
        plt.plot(test_time_array[5:], predicted_cartesian_for_rf[i][5:, 1], label='RF Y Position')
        plt.plot(test_time_array[5:], predicted_cartesian_for_rf[i][5:, 2], label='RF Z Position')

        plt.xlabel('Time (s)')
        plt.ylabel('Cartesian Position (m)')
        plt.title('Predicted Cartesian Positions Over Time Comparison')
        plt.legend()
        plt.grid(True)
        save_path = f"Figures/task3.1/Comparison Graphs/Test {i + 1}/2D-graph.png"
        try:
            plt.savefig(save_path, dpi=300)
        except:
            print("Directory doesnt exist, most likely need to add extra test folders.")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_nn[goal_positions.index(goal_position)][5:, 0], test_time_array[5:]), label='NN X Position')
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_nn[goal_positions.index(goal_position)][5:, 1], test_time_array[5:]), label='NN Y Position')
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_nn[i][5:, 2], test_time_array[5:]), label='NN Z Position')
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_rf[i][5:, 0], test_time_array[5:]), label='RF X Position')
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_rf[i][5:, 1], test_time_array[5:]), label='RF Y Position')
        plt.plot(test_time_array[5:], np.gradient(predicted_cartesian_for_rf[i][5:, 2], test_time_array[5:]), label='RF Z Position')

        plt.xlabel('Time (s)')
        plt.ylabel('Cartesian Velocity (m/s)')
        plt.title('Predicted Cartesian Velocity Over Time Comparison')
        plt.legend()
        plt.grid(True)
        save_path = f"Figures/task3.1/Comparison Graphs/Test {i + 1}/2D-graph-velocities.png"
        try:
            plt.savefig(save_path, dpi=300)
        except:
            print("Directory doesnt exist, most likely need to add extra test folders.")
        plt.show()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(predicted_cartesian_for_nn[i][5:, 0], predicted_cartesian_for_nn[i][5:, 1], predicted_cartesian_for_nn[i][5:, 2], label='Predicted Trajectory for Neural Network')
        ax.plot(predicted_cartesian_for_rf[i][5:, 0], predicted_cartesian_for_rf[i][5:, 1], predicted_cartesian_for_rf[i][5:, 2], label='Predicted Trajectory for Random Forest')
        ax.scatter(goal_position[0], goal_position[1], goal_position[2], color='red', label=f'Goal: ({goal_position[0]:.3f}, {goal_position[1]:.3f}, {goal_position[2]:.3f})')
        ax.scatter(predicted_cartesian_for_nn[i][-1, 0], predicted_cartesian_for_nn[i][-1, 1], predicted_cartesian_for_nn[i][-1, 2], color='green', label=f"Neural Network End: ({predicted_cartesian_for_nn[i][-1, 0]:.3f}, {predicted_cartesian_for_nn[i][-1, 1]:.3f}, {predicted_cartesian_for_nn[i][-1, 2]:.3f})")
        ax.scatter(predicted_cartesian_for_rf[i][-1, 0], predicted_cartesian_for_rf[i][-1, 1], predicted_cartesian_for_rf[i][-1, 2], color='green', label=f"Random Forest End: ({predicted_cartesian_for_rf[i][-1, 0]:.3f}, {predicted_cartesian_for_rf[i][-1, 1]:.3f}, {predicted_cartesian_for_rf[i][-1, 2]:.3f})")
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('Predicted Cartesian Trajectory')
        plt.legend(loc=9)
        save_path = f"Figures/task3.1/Comparison Graphs/Test {i + 1}/3D-graph.png"
        try:
            plt.savefig(save_path, dpi=300)
        except:
            print("Directory doesnt exist, most likely need to add extra test folders.")
        plt.show()

if __name__ == '__main__':
    main()
