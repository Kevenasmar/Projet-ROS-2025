import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.subscription  # prevent unused variable warning

        
        self.br = CvBridge()
        
    # def line_detection(self,data):
    #     current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
    #     frame_binary = cv2.imread(current_frame,cv2.IMREAD_GRAYSCALE)
        
    
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        # Convert BGR to HSV (best for color filtering)
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # --- GREEN MASK ---
        lower_green = (50, 60, 20)
        upper_green = (110, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # --- RED MASK (requires 2 ranges because red wraps around hue space) ---
        lower_red1 = (0, 70, 70)
        upper_red1 = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = (170, 70, 70)
        upper_red2 = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine both red masks
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Combine green and red masks
        mask = cv2.bitwise_or(mask_green, mask_red)

        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(current_frame, current_frame, mask=mask)
        cv2.imshow("camera", masked_frame)
        cv2.waitKey(1)
        
        


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()