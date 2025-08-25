import rclpy
from rclpy.node import Node

from rl_interfaces.msg import MotionCommand
from sensor_msgs.msg import JointState

import socket


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('training_ros2socket')
        # The actions are the torque for the motors, published in the structure accepted by the MD80 motor.
        self.publisher_ = self.create_publisher(MotionCommand, '/md80/motion_command', 10)
        self.motor_id = int(input("Enter the motor ID to control: "))  # Get motor ID from user input
        self.port = int(input("Enter port to connect to (by default 9998): "))  # Get port from user input

        self.subscription = self.create_subscription(
            JointState,
            '/md80/joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.actionFlag = False
        self.msgMd80ExoHz = 100
        self.msgMd80ExoVel = 0
        

    def listener_callback(self, msg):
        #JointState has a string[] name, double[] position, double[] velocity and double[] effort
        self.get_logger().info('Exo data: "%s"' % msg.position)
        print("Starting client")

        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if not self.actionFlag:
            try:
                self.socket_client.connect(('localhost', self.port))
                response = self.socket_client.recv(1024)
                print(f"Action from server: {response.decode()}")

                act = float(response.decode())  # Assuming the server sends a float action

                msg_pub = MotionCommand()   # We create a MotionCommand message
                msg_pub.drive_ids = [self.motor_id]  # Use the motor ID from user input
                msg_pub.target_torque = [act]   # We introduce the torque obtained from the RL into the message
                # Publish the message
                self.publisher_.publish(msg_pub)       
                # Log the published message to the console
                self.get_logger().info(f'Publishing torque: "{msg_pub.target_torque}"')

                self.actionFlag = True

            except Exception as e:
                print(f"Error occurred: {e}")
            finally:
                self.socket_client.close()
                print("Connection closed.")
        
        else:
            try:
                self.socket_client.connect(('localhost', self.port))
                print(f"Connected to server on port {self.port}")

                position = msg.position[0]
                acceleration = (msg.velocity[0] - self.msgMd80ExoVel) /(2*(1/self.msgMd80ExoHz))
                self.msgMd80ExoVel = msg.velocity[0]
                velocity = msg.velocity[0]
                torque = msg.effort[0]

                message = f"{position}, {velocity}, {acceleration}, {torque}"
                self.socket_client.sendall(message.encode())
                print("Observation sent!")
                self.actionFlag = False

            except Exception as e:
                print(f"Error occurred: {e}")
            finally:
                self.socket_client.close()
                print("Connection closed.")


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()