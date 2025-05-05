import rclpy
from rclpy.node import Node
import threading
import click
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan
import numpy as np
import cv2
import math

class MyTeleopNode(Node):

    def __init__(self):
        super().__init__('turtlebot_teleop', allow_undeclared_parameters=False, automatically_declare_parameters_from_overrides=True)
        
        # self.subscription = self.create_subscription(
        #     CompressedImage,
        #     'camera/image_raw/compressed',
        #     self.listener_callback,
        #     10
        # )
        
        # self.subscription = self.create_subscription(
        #     LaserScan,
        #     '/scan',
        #     self.laser_callback,
        #     10
        # )
        
        #self.subscription  # prevent unused variable warning
        
        # Define the key mappings
        self.keycode = {'\x1b[A': 'up', '\x1b[B': 'down',
                        '\x1b[C': 'right', '\x1b[D': 'left', 's': 'stop', 'q': 'quit'}
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10) #ajouter /turtle1 pour remettre commandes sur la tortue

        # Declare and Retrieve Parameters
        self.declare_parameter("linear_scale", 0.1)
        self.declare_parameter("angular_scale", 1.0)

        self.linear_scale = self.get_parameter("linear_scale").value
        self.angular_scale = self.get_parameter("angular_scale").value

        self.get_logger().info(f"Linear Scale: {self.linear_scale}")
        self.get_logger().info(f"Angular Scale: {self.angular_scale}")
        
        #self.emergency_stop = False

    def run(self):
        """Function to capture keyboard input and publish Twist messages."""
        try:
            mykey = click.getchar()
            char = self.keycode.get(mykey, None)

            command = Twist()
            if char == 'up':
                command.linear.x = self.linear_scale
                command.angular.z = 0.0

            elif char == 'down':
                command.linear.x = -self.linear_scale
                command.angular.z = 0.0

            elif char == 'right':
                command.linear.x = 0.0
                command.angular.z = -self.angular_scale

            elif char == 'left':
                command.linear.x = 0.0
                command.angular.z = self.angular_scale

            elif char == 'stop':
                command.linear.x = 0.0
                command.angular.z = 0.0

            elif char == 'quit':
                self.get_logger().info("Exiting teleop node.")
                rclpy.shutdown()
                exit(0)

            # Publish the command only if a valid key is pressed
            if char is not None:
                self.publisher.publish(command)
        
        except Exception as e:
            self.get_logger().error(f"Error in teleop: {e}")
            
    def listener_callback(self,msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn("Failed to decode compressed image")
            return
        
        height, width, _ = image.shape
        cropped_frame = image[height // 2 + 25 :,:]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        lower_green = (40, 40, 40)
        upper_green = (90, 255, 255)    
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        lower_red1 = (0, 50, 50)
        upper_red1 = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = (170, 50, 50)
        upper_red2 = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        lower_blue = (100, 150, 150)  # H, S, V
        upper_blue = (130, 255, 255)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # You can adjust size
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        filtered_pure = np.zeros_like(cropped_frame)
        filtered_pure[mask_green > 0] = (0, 255, 0)
        filtered_pure[mask_red > 0] = (0, 0, 255)
        filtered_pure[mask_blue > 0] = (255, 0, 0)

        gray = cv2.cvtColor(filtered_pure, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filtered_pure, contours, -1, (255, 255, 255), 1)
        
        #cv2.imshow("Filtered Pure (rouge/vert/noir uniquement)", filtered_pure)
        #cv2.waitKey(1)
        
    # def laser_callback(self, msg):
    #     np_array = np.array(msg.ranges)
    #     front = np.mean(np.concatenate((np_array[-10:], np_array[:10])))
    #     left = np.mean(np_array[80:100])
    #     back = np.mean(np_array[170:190])
    #     right = np.mean(np_array[260:280])  
    #     distances = [front,left,back,right]
    #     command = Twist()
    #     if min(distances) < 0.5:
    #         self.emergency_stop = True
    #         command.linear.x = 0.0
    #         command.angular.z = 0.0
    #         self.publisher.publish(command)
    #         self.get_logger().warn("⚠️ Obstacle détecté : arrêt d'urgence !")
    #     else:
    #         self.emergency_stop = False


def main(args=None):
    rclpy.init(args=args)
    teleop_node = MyTeleopNode()

    #Start `rclpy.spin()` in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(teleop_node,), daemon=True)
    thread.start()

    try:
        #Keep the keyboard input in the main thread
        while True:
            teleop_node.run()
    except KeyboardInterrupt:
        teleop_node.get_logger().info("Keyboard Interrupt detected, shutting down.")
    finally:
        teleop_node.destroy_node()
        rclpy.shutdown()
        thread.join()  # Ensure the ROS2 thread stops before exiting

if __name__ == '__main__':
    main()
