#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import math
import cv2

class LineFollowerAvoider(Node):
    def __init__(self):
        super().__init__('line_follower_avoider')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(CompressedImage, 'camera/image_raw/compressed', self.image_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.obstacle_threshold = 0.25
        self.threshold_distance = 25
        self.obstacle_right = False
        self.obstacle_left = False
        self.threshold_distance_green = 330
        self.vy_target = 0.7
        self.turn_direction = "left" #or "right"
        self.waiting_for_start = False
        self.latest_twist_from_camera = None
        self.last_image_time = self.get_clock().now()

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
        front_left = valid_ranges[:15]
        front_right = valid_ranges[-15:]
        self.get_logger().info(f"Distance lidar min front left: {np.min(front_left)}")
        self.get_logger().info(f"Distance lidar min front right: {np.min(front_right)}")
        self.obstacle_right = np.any(front_right < self.obstacle_threshold)
        self.obstacle_left = np.any(front_left < self.obstacle_threshold)

    def clean_mask(self, mask, kernel_size=(5, 5), min_area=200):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(cleaned_mask, [cnt], -1, 255, -1)
        return cleaned_mask

    def get_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
    
    def get_closest_point(self, mask):
        coords = cv2.findNonZero(mask)
        if coords is not None:
            bottommost = max(coords, key=lambda pt: pt[0][1])
            x = bottommost[0][0]
            y = bottommost[0][1]
            return x,y
        return None

    def get_tangent_components_and_draw(self, mask, centroid, color, frame):
        if centroid is None:
            return None, None

        cx, cy = centroid
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) < 10:
            return None, None

        distances = np.linalg.norm(coords - np.array([cy, cx]), axis=1)
        close_points = coords[distances < 20]
        if len(close_points) < 2:
            return None, None

        points_xy = np.array([[x, y] for y, x in close_points])
        [vx, vy, _, _] = cv2.fitLine(points_xy, cv2.DIST_L2, 0, 0.01, 0.01)

        if vy > 0:
            vx = -vx
            vy = -vy

        angle_rad = math.atan2(vy, vx)
        angle_deg = np.degrees(angle_rad) % 180

        length = 50
        x1 = int(cx + length * vx)
        y1 = int(cy + length * vy)
        cv2.line(frame, (cx, cy), (x1, y1), color, 2)

        cv2.putText(frame, f"{angle_deg:.1f} deg", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"cos: {vx.item():.2f}", (cx + 10, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"sin: {vy.item():.2f}", (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return vx.item(), vy.item()

    def image_callback(self, msg):
        now = self.get_clock().now()
        if (now - self.last_image_time).nanoseconds < 1e8:  # 10 FPS throttle
            return
        self.last_image_time = now

        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn("Failed to decode compressed image")
            return

        height, width, _ = image.shape
        cropped = image[height // 2 + 25:, :]
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Masks
        mask_green = self.clean_mask(cv2.inRange(hsv, (40, 40, 40), (90, 255, 255)))
        mask_red = self.clean_mask(cv2.bitwise_or(
            cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        ))

        # Visualization
        display = np.zeros_like(cropped)
        display[mask_green > 0] = (0, 255, 0)
        display[mask_red > 0] = (0, 0, 255)

        # Centroids
        green_c = self.get_centroid(mask_green)
        red_c = self.get_centroid(mask_red)
        
        if green_c:
            cv2.circle(display, green_c, 5, (0, 255, 0), -1)
        if red_c:
            cv2.circle(display, red_c, 5, (0, 0, 255), -1)
            

        green_vx, green_vy = self.get_tangent_components_and_draw(mask_green, green_c, (0, 255, 0), display)
        red_vx, red_vy = self.get_tangent_components_and_draw(mask_red, red_c, (0, 0, 255), display)

        twist = Twist()
        center_x = cropped.shape[1] // 2
        
        green_point = self.get_closest_point(mask_green)
        red_point = self.get_closest_point(mask_red)

        if green_point is not None:
            green_x, green_y = green_point
            # do something with green_x, green_y

        if red_point is not None:
            red_x, red_y = red_point
            # do something with red_x, red_y

        if green_vx is not None and red_vx is not None:
            if abs(green_x-red_x) < 10 : 
                self.get_logger().info(f"Ligne rouge et verte très proche : {abs(green_x-red_x)}")
                twist.linear.x = 0.0
                twist.angular.z = 0.4 if self.turn_direction == 'left' else -1.7
                
            else:
                speed = 0.5 * (abs(green_vx) + abs(red_vx) + abs(green_vy) + abs(red_vy)) / 2
                twist.linear.x = min(speed, 0.07)
                twist.angular.z = 0.0
                self.get_logger().info("2 lignes détectées : avance proportionnelle à la tangente")

        elif red_vx is not None:
            if (red_x and red_y) is not None and red_c[0] > display.shape[1] - self.threshold_distance:
                twist.linear.x = 0.03
                twist.angular.z = 0.0
                self.get_logger().info("Rouge loin → avance vers seuil")
            if red_x < display.shape[1] - self.threshold_distance:
                twist.linear.x = 0.0
                twist.angular.z = 0.3
                self.get_logger().info("Rouge proche mais non alignée → tourne à gauche")
            else:
                twist.linear.x = 0.03
                twist.angular.z = 0.0
                self.get_logger().info("Rouge proche et alignée → avance")

        elif green_vx is not None:
            if (green_x and green_y) is not None and green_c[0] < display.shape[1] - self.threshold_distance_green:
                twist.linear.x = 0.03
                twist.angular.z = 0.0
                self.get_logger().info("Vert loin → avance vers seuil")
            elif green_c[0] > display.shape[1] - self.threshold_distance_green:
                twist.linear.x = 0.0
                twist.angular.z = -0.3
                self.get_logger().info("Vert proche mais non alignée → tourne à droite")
            else:
                twist.linear.x = 0.03
                twist.angular.z = 0.0
                self.get_logger().info("Vert proche et alignée → avance")

        else:
            twist.linear.x = 0.025
            twist.angular.z = 0.0
            #self.get_logger().info("Aucune ligne détectée → avance pour en trouver")
        
        self.latest_twist_from_camera = twist

        # Show what the robot sees
        cv2.imshow("Line View", display)
        cv2.waitKey(1)

    def control_loop(self):
        twist = Twist()
        if self.obstacle_left or self.obstacle_right:
            twist.linear.x = 0.0
            twist.angular.z = -0.2 if self.obstacle_left else 0.2
            self.get_logger().warn("🚧 Avoiding obstacle")
        elif self.latest_twist_from_camera:
            twist = self.latest_twist_from_camera
        else:
            twist.linear.x = 0.05
            twist.angular.z = 0.0
            #self.get_logger().info("🕳️ Default move")

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerAvoider()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
