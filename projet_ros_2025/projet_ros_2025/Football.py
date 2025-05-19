import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import cv2
import numpy as np

class BallTracker(Node):
    def __init__(self):
        super().__init__('ball_tracker')
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = 'align_ball'
        self.cx_ball = None
        self.cx_goal = None
        self.height = None
        self.width = None

    def clean_mask(self, mask, kernel_size=(3, 3), min_area=200):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(cleaned_mask, [cnt], -1, 255, -1)
        return cleaned_mask

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.height, self.width, _ = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect yellow ball
        lower_yellow = np.array([20, 80, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = self.clean_mask(cv2.inRange(hsv, lower_yellow, upper_yellow))

        M = cv2.moments(mask_yellow)
        if M['m00'] > 0:
            self.cx_ball = int(M['m10'] / M['m00'])
        else:
            self.cx_ball = None

        # Detect red goal posts
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red = self.clean_mask(
            cv2.bitwise_or(
                cv2.inRange(hsv, lower_red1, upper_red1),
                cv2.inRange(hsv, lower_red2, upper_red2)
            )
        )

        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                centers.append(int(M['m10'] / M['m00']))
        self.cx_goal = int(sum(centers) / len(centers)) if len(centers) == 2 else None

        # Visualization
        overlay = cv2.bitwise_and(image, image, mask=mask_yellow | mask_red)
        if self.cx_ball:
            cv2.circle(overlay, (self.cx_ball, self.height // 2), 5, (0, 255, 255), -1)
        if self.cx_goal:
            cv2.circle(overlay, (self.cx_goal, self.height // 2), 5, (0, 255, 0), -1)
        cv2.putText(overlay, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Debug View", overlay)
        cv2.waitKey(1)

    def control_loop(self):
        twist = Twist()
        center_x = self.width // 2 if self.width else 320

        if self.state == 'align_ball':
            if self.cx_ball is not None:
                error = self.cx_ball - center_x
                if abs(error) > 15:
                    twist.linear.x = 0.01
                    twist.angular.z = -0.002 * error
                    self.get_logger().info(f"Aligning with ball: error={error}")
                else:
                    self.get_logger().info("Ball aligned → driving toward ball")
                    self.state = 'drive_to_ball'
            else:
                twist.angular.z = 0.2
                self.get_logger().info("Searching for ball...")

        elif self.state == 'drive_to_ball':
            if self.cx_ball is not None:
                error = self.cx_ball - center_x
                if abs(error) > 15:
                    self.state = 'align_ball'
                    self.get_logger().info(f"Ball misaligned during drive → returning to align (error: {error})")
                else:
                    twist.linear.x = 0.1
                    self.get_logger().info("Driving straight to ball...")
            else:
                twist = Twist()
                self.state = 'search_goal'
                self.get_logger().info("Ball lost → transitioning to goal search")


        elif self.state == 'search_goal':
            if self.cx_goal is not None:
                twist = Twist()
                self.state = 'align_goal'
                self.get_logger().info("Goal found → aligning")
            else:
                twist.angular.z = 0.3
                self.get_logger().info("Searching for goal...")

        elif self.state == 'align_goal':
            if self.cx_goal is not None:
                error = self.cx_goal - center_x
                if abs(error) > 15:
                    twist.angular.z = -0.002 * error
                    self.get_logger().info(f"Aligning with goal: error={error}")
                else:
                    twist = Twist()
                    self.state = 'drive_to_goal'
                    self.get_logger().info("Goal aligned → moving forward")
            else:
                self.state = 'search_goal'
                self.get_logger().info("Goal lost → re-searching")

        elif self.state == 'drive_to_goal':
            twist.linear.x = 0.1
            self.get_logger().info("Driving straight to goal...")

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = BallTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
