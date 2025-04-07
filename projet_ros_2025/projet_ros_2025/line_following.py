import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import click
import cv2
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.br = CvBridge()
        
    def get_centroid_from_band(self,mask, y_start, y_end):
            band = mask[y_start:y_end, :]
            M = cv2.moments(band)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                return cx
            return None
    
    def get_closest_y(self, mask):
        """
        Returns the largest Y value (closest red pixel to the bottom).
        """
        coords = cv2.findNonZero(mask)
        if coords is not None:
            bottommost = max(coords, key=lambda pt: pt[0][1])
            return bottommost[0][1]
        return None

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
        height, width, _ = current_frame.shape

        cropped_frame = current_frame[height//2 -100:, :]
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
        
        # Use only the bottom 20 pixels of cropped frame
        band_height = 200
        y_start = cropped_frame.shape[0] - band_height
        y_end = cropped_frame.shape[0]

        green_cx = self.get_centroid_from_band(mask_green, y_start, y_end)
        red_cx = self.get_centroid_from_band(mask_red, y_start, y_end)

        image_center = cropped_frame.shape[1] // 2
        msg = Twist()

        # Constants for control
        Kp = 0.002  # Angular gain
        max_speed = 0.1
        K_speed_drop = 0.001  # Speed penalty per pixel of error

        if green_cx is not None and red_cx is not None:
            # Both lines visible → standard midpoint logic
            mid_x = (green_cx + red_cx) // 2
            error = mid_x - image_center
            msg.linear.x = 0.1
            msg.angular.z = -Kp * error

            cv2.circle(masked_frame, (green_cx, 100), 5, (0, 255, 0), -1)
            cv2.circle(masked_frame, (red_cx, 100), 5, (0, 0, 255), -1)
            cv2.circle(masked_frame, (mid_x, 100), 5, (255, 0, 0), -1)

        elif red_cx is not None:
            # Only red visible → go straight if far, turn if close
            closest_red_y = self.get_closest_y(mask_red)
            self.get_logger().info(f"Closest red Y: {closest_red_y}")

            if closest_red_y is not None and closest_red_y < cropped_frame.shape[0] - 40:
                # Red is far away → go straight
                msg.linear.x = 0.1
                msg.angular.z = 0.0
                self.get_logger().info("Red line far → going straight.")
            else:
                # Red is close → turn in place
                error = red_cx - image_center
                msg.angular.z = 0.6
                msg.linear.x = 0.0
                self.get_logger().info("Red line close → turning.")

            cv2.circle(masked_frame, (red_cx, 100), 5, (0, 0, 255), -1)

        elif green_cx is not None:
            # Only green visible → go straight if far, turn if close
            closest_green_y = self.get_closest_y(mask_green)
            self.get_logger().info(f"Closest green Y: {closest_green_y}")

            if closest_green_y is not None and closest_green_y < cropped_frame.shape[0] - 80:
                # Red is far away → go straight
                msg.linear.x = 0.1
                msg.angular.z = 0.0
                self.get_logger().info("Green line far → going straight.")
            else:
                # Green is close → turn in place
                error = green_cx - image_center
                msg.angular.z = -0.6
                msg.linear.x = 0.
                
            cv2.circle(masked_frame, (green_cx, 100), 5, (0, 255, 0), -1)

        else:
            # No lines → stop
            msg.linear.x = 0.005
            msg.angular.z = 0.005
            self.get_logger().warn("No lines detected. Robot stopping.")

        # Publish and show
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