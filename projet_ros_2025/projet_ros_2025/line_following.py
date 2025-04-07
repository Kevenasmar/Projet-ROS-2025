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
    #     # Convert BGR to HSV (best for color filtering)
    #     hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    #     # --- GREEN MASK ---
    #     lower_green = (50, 60, 20)
    #     upper_green = (110, 255, 255)
    #     mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #     # --- RED MASK (requires 2 ranges because red wraps around hue space) ---
    #     lower_red1 = (0, 70, 70)
    #     upper_red1 = (10, 255, 255)
    #     mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #     lower_red2 = (170, 70, 70)
    #     upper_red2 = (180, 255, 255)
    #     mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    #     # Combine both red masks
    #     mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
    
    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        height, width, _ = current_frame.shape

        cropped_frame = current_frame[height//2:, :]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        # Define green and red masks
        lower_green = (50, 60, 20)
        upper_green = (110, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        lower_red1 = (0, 70, 70)
        upper_red1 = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = (170, 70, 70)
        upper_red2 = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask = cv2.bitwise_or(mask_green, mask_red)
        masked_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)

        # Find centroids
        M_green = cv2.moments(mask_green)
        M_red = cv2.moments(mask_red)
        green_cx = red_cx = None

        if M_green["m00"] > 0:
            green_cx = int(M_green["m10"] / M_green["m00"])
        if M_red["m00"] > 0:
            red_cx = int(M_red["m10"] / M_red["m00"])

        if green_cx is not None and red_cx is not None:
            mid_x = int((green_cx + red_cx) / 2)

            # Draw visual markers
            cv2.circle(masked_frame, (green_cx, 100), 5, (0, 255, 0), -1)
            cv2.circle(masked_frame, (red_cx, 100), 5, (0, 0, 255), -1)
            cv2.circle(masked_frame, (mid_x, 100), 5, (255, 0, 0), -1)

            # --- Control ---
            image_center = cropped_frame.shape[1] // 2
            error = mid_x - image_center

            # Create and publish Twist message
            msg = Twist()
            msg.linear.x = 0.1  # Constant forward speed
            msg.angular.z = -0.002 * error  # Proportional control for turning (tune gain!)

            self.publisher.publish(msg)

        else:
            # If line not detected properly, stop
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.publisher.publish(msg)

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