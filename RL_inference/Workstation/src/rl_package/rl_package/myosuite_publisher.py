# We import the necessary packages
import rclpy
from rclpy.node import Node

# Note: the Float32MultiArray may not work on ROS2 Humble, as ROS2 developers don't like the use of generic messages types and is supposed to be depreciated in Foxy.
from std_msgs.msg import Float32MultiArray
from rl_interfaces.msg import MotionCommand

import numpy
import sys \

print(sys.version)
sys.path.append('/home/nair-group/miniconda/envs/ros2rl/lib/python3.10/site-packages')
print(sys.version)

from myosuite.utils import gym


class MinimalPublisher(Node):
    env = gym.make('ExoLegSpasticityFlexoExtEnv-v0')

    def __init__(self):
        # We create the MyoSuite environment, the subscriber to listen to the actions and the publisher to send the observations
        super().__init__('myosuite')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/env/myosuite_obs', 10)

        self.motor_id = int(input("Enter the motor ID to control: "))  # Get motor ID from user input

        env = gym.make('ExoLegSpasticityFlexoExtEnv-v0')
        obs = self.env.reset()
        self.get_logger().info('MyoSuite environment created and reset.')

        msg_pub = Float32MultiArray() # The published message is an array of floats
        msg_pub.data = obs[0].tolist() # We convert the numpy array into a Python list and save it into the message
        self.publisher_.publish(msg_pub)
        self.get_logger().info('Publishing MyoSuite observations: "%s"' % msg_pub.data)

        self.subscription = self.create_subscription(
            MotionCommand,              # Message type used by the publisher
            '/md80/motion_command',     # topic name
            self.listener_callback,     # Method called each time a message is received
            10)                         # Queue size for incoming messages
        self.subscription  # Prevent unused variable warning


    def listener_callback(self, msg):
        # We get the action, perform a step in the MyoSuite environment and publish the obtained observations
        # The action will have 41 floats and the observations will have 91 floats, probably as a numpy array

        # Log the received message to the console
        self.get_logger().info(f'Torque received: "{msg.target_torque}"')

        if(msg.drive_ids[0] == self.motor_id):
            self.get_logger().info(f'Receiving message for motor with ID: {self.motor_id}')

            # Here we would do the env.step to obtain the observations
            action = numpy.zeros(41)
            action[0] = msg.target_torque[0]  # Assuming the first element is the exo torque

            # obs = self.env.step(self.env.action_space.sample())
            obs = self.env.step(action)
            print(self.env.action_space.sample())
            print(type(self.env.action_space.sample()))
            self.get_logger().info('Step performed in MyoSuite environment.')
            print(obs)
            print(type(obs))
            print(len(obs))
            print(type(obs[0]))
            print(len(obs[0]))

            # array = numpy.array([0.0,msg.target_torque[0],2.0,3.0,4.0,5.0])
            array = obs[0]
            msg_pub = Float32MultiArray() # The published message is an array of floats
            msg_pub.data = array.tolist() # We convert the numpy array into a Python list and save it into the message
            #msg_pub.data = obs[0].tolist()
            self.publisher_.publish(msg_pub)
            self.get_logger().info('Publishing MyoSuite observations: "%s"' % msg_pub.data)



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher() # We create the node

    try:
        # Run the node until interrupted
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Gracefully handle shutdown when Ctrl+C is pressed
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        #minimal_publisher.close_env()
        minimal_publisher.env.close()  # Close the MyoSuite environment
        minimal_publisher.get_logger().info('MyoSuite environment closed.')
        minimal_publisher.destroy_node()
        rclpy.shutdown()

# Entry point of the script
if __name__ == '__main__':
    main()
