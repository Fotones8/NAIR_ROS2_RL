# We import the necessary packages
import rclpy
from rclpy.node import Node

# Note: the Float32MultiArray may not work on ROS2 Humble, as ROS2 developers don't like the use of generic messages types and is supposed to be depreciated in Foxy.
from std_msgs.msg import Float32MultiArray
from rl_interfaces.msg import MotionCommand

import numpy



class MinimalPublisher(Node):


    def __init__(self):
        # We create the publisher for the RL actions and the subscriber for the RL observations
        super().__init__('rl_node')
        # The actions are the torque for the motors, published in the structure accepted by the MD80 motor.
        self.publisher_ = self.create_publisher(MotionCommand, '/md80/motion_command', 10)

        self.i = 0

        self.subscription = self.create_subscription(
            Float32MultiArray,          # Message type used by the publisher
            '/rl/observations',         # topic name
            self.listener_callback,     # Method called each time a message is received
            10)                         # Queue size for incoming messages
        self.subscription  # Prevent unused variable warning

        # We send an initial message to start the feedback loop
        print('Starting the system')
        msg_pub = MotionCommand()
        msg_pub.target_torque = [0.0, 0.0]        
        # Publish the message
        self.publisher_.publish(msg_pub)


    def listener_callback(self, msg):
        # Each time we receive the observations, we must pass it to the RL to obtain the actions, and then we must publish the actions
        observations = msg.data     # The observations are a float array found in msg.data
        self.get_logger().info(f'Receiving observations: "{observations}"')

        # Here we would pass the observations to the RL and obtain the actions
        action = 0.0+self.i

        # MotionCommand has uint32[] drive_ids, float32[] target_position, float32[] target_velocity, float32[] target_torque
        msg_pub = MotionCommand()   # We create a MotionCommand message
        msg_pub.target_torque = [action, 0.0]   # We introduce the torque obtained from the RL into the message
        # Publish the message
        self.publisher_.publish(msg_pub)       
        # Log the published message to the console
        self.get_logger().info(f'Publishing torque: "{msg_pub.target_torque}"')

        self.i += 1


def main(args=None):
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)
    
    # Create an instance of the MinimalPublisher node
    minimal_publisher = MinimalPublisher()

    try:
        # Run the node until interrupted
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Gracefully handle shutdown when Ctrl+C is pressed
        pass
    finally:
        # Destroy the node and shutdown the ROS2 Python client library
        minimal_publisher.destroy_node()
        rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()
