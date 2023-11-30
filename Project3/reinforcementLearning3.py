import rospy
import numpy as np
import random
import pickle
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Twist, PoseStamped

class CentralizedQLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            #return random.choice(range(self.action_size))
            return random.choice(range(0,8))
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * target

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RobotController:
    def __init__(self):
        self.robot1_lidar_data=None
        self.robot2_lidar_data=None
        self.initial_robot1_pose = None
        self.initial_robot2_pose = None

        self.prev_robot1_distance = 0.0
        self.prev_robot2_distance = 0.0
        self.cmd1 = Twist()
        self.cmd2 = Twist()
        self.discrete_actions = [(4.9, 0), (4.9, 0.5), (4.9, -0.5), (4.9, 1), (4.9, -1), (4.0, 1.5), (4.0, -1.5), (0, 0)]
        self.state_size = 4  # Adjust based on your state representation
        self.action_size = len(self.discrete_actions)

        central_state_size = self.state_size*self.state_size  # Combining states of both robots
        central_action_size = self.action_size*self.action_size

        self.central_q_agent = CentralizedQLearning(central_state_size, central_action_size)
        
        # Define robot goals and initial robot poses
        self.robot1_goal = (8.27, -4.64, 1.19, 0.3)
        self.robot2_goal = (5.51, 2.17, 1.14, 0.3)
        self.robot1_pose = PoseStamped()
        self.robot2_pose = PoseStamped()
        self.robot1_cmd_pub = rospy.Publisher('warty1/cmd_vel', Twist, queue_size=10)
        self.robot2_cmd_pub = rospy.Publisher('warty2/cmd_vel', Twist, queue_size=10)

        rospy.Subscriber('warty1/lidar_points', PointCloud, self.robot1_lidar_callback)
        rospy.Subscriber('warty2/lidar_points', PointCloud, self.robot2_lidar_callback)
        rospy.Subscriber('warty1/pose', PoseStamped, self.robot1_pose_callback)
        rospy.Subscriber('warty2/pose', PoseStamped, self.robot2_pose_callback)
        self.current_step=0
    
    def robot1_lidar_callback(self, msg):
            # Process lidar data for robot 1
        self.robot1_lidar_data = msg

    def robot2_lidar_callback(self, msg):
            # Process lidar data for robot 2
        self.robot2_lidar_data = msg
            
    def robot1_pose_callback(self,pose):
        self.robot1_pose=pose

    def robot2_pose_callback(self,pose):
        self.robot2_pose = pose

    def calculate_distance_to_goal(self, current_pose, goal):
        return np.sqrt((current_pose.pose.position.x - goal[0]) ** 2 +
                   (current_pose.pose.position.y - goal[1]) ** 2 +
                   (current_pose.pose.position.z - goal[2]) ** 2)


    def get_yaw_from_orientation(self, orientation):
        # Assuming orientation is a Quaternion with x, y, z, w fields
        return np.arctan2(2.0 * (orientation.y * orientation.z + orientation.x * orientation.w),
                          orientation.w**2 - orientation.x**2 - orientation.y**2 + orientation.z**2)

    def get_robot1_yaw(self):
        # Access the orientation information from the pose message of robot 1
        return self.get_yaw_from_orientation(self.robot1_pose.pose.orientation)

    def get_robot2_yaw(self):
        # Access the orientation information from the pose message of robot 2
        return self.get_yaw_from_orientation(self.robot2_pose.pose.orientation)
    def state_to_index(self, state):
        return hash(state)% self.state_size
    
    def get_state(self, robot_number):
        if robot_number == 1:
            robot_pose = (
                self.robot1_pose.pose.position.x,
                self.robot1_pose.pose.position.y,
                self.get_robot1_yaw()
            )
            robot_lidar_distances = self.process_lidar_data(self.robot1_lidar_data)
            robot_goal_state = (
                self.robot1_goal[0],
                self.robot1_goal[1]
            )
        else:
            robot_pose = (
                self.robot2_pose.pose.position.x,
                self.robot2_pose.pose.position.y,
                self.get_robot2_yaw()
            )
            robot_lidar_distances = self.process_lidar_data(self.robot2_lidar_data)
            robot_goal_state = (
                self.robot2_goal[0],
                self.robot2_goal[1]
            )

        # Combine the state information for the selected robot
        state = robot_pose + tuple(robot_lidar_distances) + robot_goal_state
        state_index = self.state_to_index(state)

        return state_index

    def process_lidar_data(self, lidar_data):
        if lidar_data is not None:
            lidar_points = np.array([(p.x, p.y, p.z) for p in lidar_data.points])
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
            lidar_distances = []

            for angle in angles:
                x = np.cos(angle)
                y = np.sin(angle)

                # Find the closest lidar point in the direction of the angle
                distances = np.linalg.norm(lidar_points[:, :2] - np.array([x, y]), axis=1)
                min_distance = np.min(distances)

                lidar_distances.append(min_distance)

            return lidar_distances
        else:
            return []

    def train_q_learning(self):
        #rate = rospy.Rate(10)  # Define the control rate (e.g., 10 Hz)
        max_episodes = 100
        episode = 0
        totalReward1= 0
        totalReward2=0

        while not rospy.is_shutdown() and episode < max_episodes:
            #if self.is_episode_done():
            #    self.reset_robots_to_initial_positions()
            #    episode+=1
            #    continue
            # Logic for training the centralized Q-learning for both robots
            state_robot1 = self.get_state(1)
            state_robot2 = self.get_state(2)
            # Action selection for both robots
            action_robot1 = self.central_q_agent.choose_action(state_robot1)
            print(action_robot1, "action robot 1")
            action_robot2 = self.central_q_agent.choose_action(state_robot2)
            print(action_robot2, "action robot 2")

            # Take actions for both robots
            self.take_action(action_robot1, action_robot2)
            if self.is_episode_done():
                self.reset()
                episode+=1
                continue
            # Calculate rewards for both robots
            reward_robot1 = self.calculate_reward_robot1(action_robot1)
            reward_robot2 = self.calculate_reward_robot2(action_robot2)
            totalReward1+=reward_robot1
            totalReward2+=reward_robot2
            # Update Q-table for both robots
            next_state_robot1 = self.get_state(1)
            next_state_robot2 = self.get_state(2)
            self.central_q_agent.update_q_table(state_robot1, action_robot1, reward_robot1, next_state_robot1)
            self.central_q_agent.update_q_table(state_robot2, action_robot2, reward_robot2, next_state_robot2)
            # Epsilon decay for both robots
            self.central_q_agent.decay_epsilon()

            # Print episode information
            print("episode", episode,"total reward 1", totalReward1,"total reward 2", totalReward2)
            #rate.sleep()

    def is_episode_done(self):
        max_steps = 100  

        robot1_distance_to_goal = self.calculate_distance_to_goal(self.robot1_pose, self.robot1_goal)
        print("robot 1 distance to goal", robot1_distance_to_goal)
        robot2_distance_to_goal = self.calculate_distance_to_goal(self.robot2_pose, self.robot2_goal)
        print("robot 2 distance to goal", robot2_distance_to_goal)
        if (robot1_distance_to_goal < 0.3 and robot2_distance_to_goal < 0.3) or self.current_step >= max_steps:
            print("episode done")
            return True
        else:
            print("Epsiode not done")
            return False
    
    def reset(self):
        #  initial positions of the robots
        initial_pose_robot1 = PoseStamped()
        initial_pose_robot1.pose.position.x = 0.0
        initial_pose_robot1.pose.position.y = 0.0
        initial_pose_robot1.pose.position.z = 0.0
        initial_pose_robot1.pose.orientation.x = -0.004984223643810371
        initial_pose_robot1.pose.orientation.y = -0.018407858368676134
        initial_pose_robot1.pose.orientation.z = 0.0006762841058762037
        initial_pose_robot1.pose.orientation.w = 0.9998179088737885

        initial_pose_robot2 = PoseStamped()
        initial_pose_robot2.pose.position.x = initial_pose_robot1.pose.position.x - 0.235
        initial_pose_robot2.pose.position.y = initial_pose_robot1.pose.position.y - 0.132
        initial_pose_robot2.pose.position.z = initial_pose_robot1.pose.position.z + 0.025

        # Orientation (initial)
        initial_pose_robot2.pose.orientation.x = 0.000
        initial_pose_robot2.pose.orientation.y = 0.000
        initial_pose_robot2.pose.orientation.z = -0.002
        initial_pose_robot2.pose.orientation.w = 1.000


    def take_action(self, action_robot1, action_robot2):
        linear_velocity_robot1, angular_velocity_robot1 = self.discrete_actions[action_robot1]
        linear_velocity_robot2, angular_velocity_robot2 = self.discrete_actions[action_robot2]

        # Execute action for robot 1
        self.cmd1.linear.x = linear_velocity_robot1
        self.cmd1.angular.z = angular_velocity_robot1
        self.robot1_cmd_pub.publish(self.cmd1)
        rospy.sleep(1)

        # Execute action for robot 2
        self.cmd2.linear.x = linear_velocity_robot2
        self.cmd2.angular.z = angular_velocity_robot2
        self.robot2_cmd_pub.publish(self.cmd2)
        rospy.sleep(1)

        # Utilize the latest available pose and lidar data for each robot
        robot1_pose = self.robot1_pose
        robot2_pose = self.robot2_pose

        # Calculate distances from the goal for each robot
        robot1_distance_to_goal = self.calculate_distance_to_goal(robot1_pose, self.robot1_goal)
        robot2_distance_to_goal = self.calculate_distance_to_goal(robot2_pose, self.robot2_goal)

        # Update distances in the QLearning instance for reward calculations for each robot
        self.prev_robot1_distance = robot1_distance_to_goal
        self.prev_robot2_distance = robot2_distance_to_goal

        print(robot1_distance_to_goal)
        print(robot2_distance_to_goal)
    

    def calculate_reward_robot1(self, action_robot1):
        # Calculate distance for robot 1 based on its pose and goal
        robot1_distance_to_goal = self.calculate_distance_to_goal(self.robot1_pose, self.robot1_goal)

        # Calculate rewards for robot 1 based on its progress towards the goal
        reward_robot1_distance = -0.1 * robot1_distance_to_goal
        reward_collision_robot1 = -10.0 if self.collision_detected(self.robot1_lidar_data) else 0.0
        reward_goal_robot1 = 0.1 * (self.prev_robot1_distance - robot1_distance_to_goal)
        reward_goal_direction_robot1 = -0.1 * (robot1_distance_to_goal - self.prev_robot1_distance)

        # Calculate total reward for robot 1
        reward_robot1 = reward_robot1_distance + reward_collision_robot1 + reward_goal_robot1 + reward_goal_direction_robot1

        # Update previous distance for robot 1 for the next step
        self.prev_robot1_distance = robot1_distance_to_goal

        return reward_robot1

    def calculate_reward_robot2(self, action_robot2):
        # Calculate distance for robot 2 based on its pose and goal
        robot2_distance_to_goal = self.calculate_distance_to_goal(self.robot2_pose, self.robot2_goal)

        # Calculate rewards for robot 2 based on its progress towards the goal
        reward_robot2_distance = -0.1 * robot2_distance_to_goal
        reward_collision_robot2 = -10.0 if self.collision_detected(self.robot2_lidar_data) else 0.0
        reward_goal_robot2 = 0.1 * (self.prev_robot2_distance - robot2_distance_to_goal)
        reward_goal_direction_robot2 = -0.1 * (robot2_distance_to_goal - self.prev_robot2_distance)

        # Calculate total reward for robot 2
        reward_robot2 = reward_robot2_distance + reward_collision_robot2 + reward_goal_robot2 + reward_goal_direction_robot2

        # Update previous distance for robot 2 for the next step
        self.prev_robot2_distance = robot2_distance_to_goal

        return reward_robot2

    
    def collision_detected(self, lidar_data):
        distance_threshold= 0.3
        lidar_distances = self.process_lidar_data(lidar_data)
        return any(distance<distance_threshold for distance in lidar_distances)

    def save_q_table(self):
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(self.central_q_agent.q_table, f)


def main_exec():
    rospy.init_node('robotcontroller')
    controller= RobotController()
    controller.train_q_learning()


if __name__ == "__main__":
    while not rospy.is_shutdown():
        main_exec()



