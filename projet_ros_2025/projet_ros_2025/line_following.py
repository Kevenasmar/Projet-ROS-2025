import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import cv2
import math

OFFSET_FRAME = 25

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        # self.scan_subscription = self.create_subscription(
        #     LaserScan,
        #     '/scan',
        #     self.laser_callback,
        #     10
        # )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.threshold_distance = 70
        self.threshold_rond_point = 20
        self.vy_target = 0.85
        self.emergency_stop = False
        self.turn_direction = 'left'  # ou 'right'
        self.bifurcation_engaged = False  # état interne
        self.speed = 0.05

    # def laser_callback(self, scan_msg):
    #     front_ranges = scan_msg.ranges[len(scan_msg.ranges)//2 - 5 : len(scan_msg.ranges)//2 + 5]
    #     front_ranges = [r for r in front_ranges if not math.isinf(r)]
    #     if front_ranges and min(front_ranges) < 0.4:
    #         self.emergency_stop = True
    #         self.get_logger().warn("⚠️ Obstacle détecté : arrêt d'urgence !")
    #     else:
    #         self.emergency_stop = False

    def get_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def get_closest_y(self, mask):
        coords = cv2.findNonZero(mask)
        if coords is not None:
            bottommost = max(coords, key=lambda pt: pt[0][1])
            return bottommost[0][1]
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

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn("Failed to decode compressed image")
            return

        height, width, _ = image.shape
        cropped_frame = image[height // 2 + OFFSET_FRAME :, 45:]
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

        lower_green = (55, 60, 20)
        upper_green = (110, 255, 255)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        lower_red1 = (0, 70, 70)
        upper_red1 = (10, 255, 255)
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = (170, 70, 70)
        upper_red2 = (180, 255, 255)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        filtered_pure = np.zeros_like(cropped_frame)
        filtered_pure[mask_green > 0] = (0, 255, 0)
        filtered_pure[mask_red > 0] = (0, 0, 255)

        gray = cv2.cvtColor(filtered_pure, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filtered_pure, contours, -1, (255, 255, 255), 1)

        green_centroid = self.get_centroid(mask_green)
        red_centroid = self.get_centroid(mask_red)

        if green_centroid:
            cv2.circle(filtered_pure, green_centroid, 5, (0, 255, 0), -1)
        if red_centroid:
            cv2.circle(filtered_pure, red_centroid, 5, (0, 0, 255), -1)

        green_vx, green_vy = self.get_tangent_components_and_draw(mask_green, green_centroid, (0, 255, 0), filtered_pure)
        red_vx, red_vy = self.get_tangent_components_and_draw(mask_red, red_centroid, (0, 0, 255), filtered_pure)

        twist = Twist()

        if self.emergency_stop:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
         # Mesurer la "quantité" de rouge et vert
        green_pixels = cv2.countNonZero(mask_green)
        red_pixels = cv2.countNonZero(mask_red)
        total_pixels = green_pixels + red_pixels

        if total_pixels == 0:
            return

        green_ratio = green_pixels / total_pixels
        red_ratio = red_pixels / total_pixels
        self.get_logger().info(f"Ratio pixels verts {green_ratio}")
        self.get_logger().info(f"Ratio pixels rouges {red_ratio}")
        # Estimer la proximité des lignes
        green_y = self.get_closest_y(mask_green)
        red_y = self.get_closest_y(mask_red)
        avg_y = min(green_y or 9999, red_y or 9999)

        if green_centroid and red_centroid:
            delta_x = abs(green_centroid[0] - red_centroid[0])
            if delta_x > 50:
                # Trop proche = probablement juste une ligne droite normale
                self.get_logger().info("🔹Ratios équilibrés mais centroïdes trop proches → PAS une bifurcation")
                self.get_logger().info(f"Delta x {delta_x}")
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                
                
                
            elif avg_y < cropped_frame.shape[0] - self.threshold_rond_point and not self.bifurcation_engaged:
                # Encore loin → avance doucement
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("🔴BIFURCATION START")
                self.get_logger().info(f"Delta x {delta_x}")
                self.bifurcation_engaged = True

            elif self.bifurcation_engaged:
                # Assez proche → on tourne
                self.get_logger().info("BIFURCATION ON")
                twist.linear.x = 0.0
                twist.angular.z = 0.4 if self.turn_direction == 'left' else -0.4
                self.get_logger().info(f"Proche de la bifurcation → tourne à {self.turn_direction}")
                if self.turn_direction == 'left':
                    if red_vx is None and green_vx is not None:  
                        self.bifurcation_engaged = False
                        self.get_logger().info(f"BIFURCATION STOP")
                if self.turn_direction == 'right': 
                    if red_vx is not None and green_vx is None:  
                        self.bifurcation_engaged = False
                        self.get_logger().info(f"BIFURCATION STOP")

            else:
                # Bifurcation détectée mais pas encore assez proche → avance doucement
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("Bifurcation détectée (déjà engagée) → avance")


        elif green_vx is not None and red_vx is not None:
            # Comportement normal avec 2 lignes
            speed = 0.5 * (abs(green_vx) + abs(red_vx) + abs(green_vy) + abs(red_vy)) / 2
            twist.linear.x = min(speed, self.speed)
            twist.angular.z = 0.0
            self.get_logger().info("2 lignes détectées : avance proportionnelle à la tangente")


        elif red_vx is not None:
            closest_y = self.get_closest_y(mask_red)
            if closest_y is not None and closest_y < cropped_frame.shape[0] - self.threshold_distance:
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("Rouge loin → avance vers seuil")
            elif abs(red_vy) < self.vy_target:
                twist.linear.x = 0.0
                twist.angular.z = 0.4
                self.get_logger().info("Rouge proche mais non alignée → tourne à gauche")
            else:
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("Rouge proche et alignée → avance")

        elif green_vx is not None:
            closest_y = self.get_closest_y(mask_green)
            if closest_y is not None and closest_y < cropped_frame.shape[0] - self.threshold_distance:
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("Vert loin → avance vers seuil")
            elif abs(green_vy) < self.vy_target:
                twist.linear.x = 0.0
                twist.angular.z = -0.4
                self.get_logger().info("Vert proche mais non alignée → tourne à droite")
            else:
                twist.linear.x = self.speed
                twist.angular.z = 0.0
                self.get_logger().info("Vert proche et alignée → avance")

        else:
            twist.linear.x = self.speed
            twist.angular.z = 0.0
            self.get_logger().info("Aucune ligne détectée → avance pour en trouver")

        self.cmd_pub.publish(twist)

        cv2.imshow("Filtered Pure (rouge/vert/noir uniquement)", filtered_pure)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageSubscriber()
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