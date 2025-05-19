import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class CorridorFollower(Node):
    def __init__(self):
        super().__init__('corridor_with_green_line')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.bridge = CvBridge()

        self.ne_values = []
        self.nw_values = []

        self.setpoint = 0.20
        self.gain = 5.0

        self.state = 'lidar_pid'
        self.image_center = 0
        self.green_error = None
        self.green_seen = False
        self.blue_detected = False
        self.final_stop_trigger = False

        self.frame_counter = 0
        self.min_frames_before_final_check = 30  # 30 ticks ≈ 3 secondes

    def lidar_callback(self, msg):
        # ranges = np.array(msg.ranges)
        # print("\n=== LIDAR RAW ===")
        # print(f"Nombre de valeurs : {len(ranges)}")
        # print(f"Exemples de valeurs brutes : {ranges[0:10]}")

        # ne_raw = np.concatenate((ranges[310:], ranges[:10]))
        # nw_raw = ranges[30:50]

        # self.ne_values = ne_raw[np.isfinite(ne_raw)]
        # self.nw_values = nw_raw[np.isfinite(nw_raw)]

        # print(f"NE (valeurs filtrées) : {self.ne_values}")
        # print(f"NW (valeurs filtrées) : {self.nw_values}")
        # print(f"Nb NE valides : {len(self.ne_values)} | Nb NW valides : {len(self.nw_values)}")
        ranges = np.array(msg.ranges)
        ne_raw = np.concatenate((ranges[-50:], ranges[:10]))
        self.ne_values = ne_raw[(np.isfinite(ne_raw)) & (ne_raw > 0.05)]

        nw_raw = ranges[30:50]
        self.nw_values = nw_raw[np.isfinite(nw_raw)]

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, w, _ = frame.shape
        self.image_center = w // 2

        # === MASQUE BLEU (déclenchement transition LIDAR -> VERT) ===
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        if self.state == 'lidar_pid':
            blue_zone = mask_blue[int(0.7 * h):, :]
            self.blue_detected = cv2.countNonZero(blue_zone) > 500

        # === MASQUE VERT (ligne à suivre) ===
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([100, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green[:int(0.5 * h), :] = 0

        self.green_seen = False
        self.green_error = None
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                self.green_error = self.image_center - cx
                self.green_seen = True
                cv2.circle(frame, (cx, int(0.9 * h)), 5, (0, 255, 0), -1)

        # === MASQUE BLEU FINAL (pour arrêt) ===
        self.final_stop_trigger = False
        if self.state == 'green_approach' and self.frame_counter > self.min_frames_before_final_check:
            h_start = int(0.75 * h)
            mask_blue_bottom = mask_blue[h_start:, :]
            w_start = int(0.4 * w)
            w_end = int(0.6 * w)
            blue_stop_zone = mask_blue_bottom[:, w_start:w_end]
            self.final_stop_trigger = cv2.countNonZero(blue_stop_zone) > 200

        # === DEBUG VIEW ===
        debug = cv2.bitwise_or(mask_green, mask_blue)
        debug_bgr = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

        status_text = f"État : {self.state} | Erreur verte : {self.green_error if self.green_error else 'None'}"
        cv2.putText(debug_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if self.final_stop_trigger:
            cv2.rectangle(debug_bgr, (w_start, h_start), (w_end, h), (255, 0, 0), 2)
            cv2.putText(debug_bgr, "ARRET FINAL BLEU", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Debug Vision", debug_bgr)
        cv2.waitKey(1)

    def control_loop(self):
        twist = Twist()

        if self.state == 'lidar_pid':
            if self.blue_detected:
                print("🟦 Ligne bleue détectée → passage au suivi de ligne verte.")
                self.state = 'green_approach'
                return

            if len(self.ne_values) > 0:
                distance = np.min(self.ne_values)
                error = self.setpoint - distance
                angular_z = self.gain * error
                angular_z = max(min(angular_z, 1.2), -1.2)
                linear_x = 0.04 if abs(angular_z) < 0.5 else 0.02

                twist.linear.x = linear_x
                twist.angular.z = angular_z

                print("\n========== PID LIDAR ==========")
                print(f"📏 Distance droite : {distance:.3f} m")
                print(f"🧮 Erreur          : {error:.3f} m")
                print(f"⚙️ Commande Z      : {angular_z:.3f}")
                print(f"🚗 Vitesse X       : {linear_x:.3f}")
                print("================================")
            else:
                print("❌ Aucune donnée LIDAR valide.")
                twist.linear.x = 0.0

        elif self.state == 'green_approach':
            self.frame_counter += 1

            if self.final_stop_trigger:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.state = 'stop'
                print("🟦 Ligne bleue finale détectée → ARRÊT.")

            elif self.green_seen and self.green_error is not None:
                twist.linear.x = 0.05
                gain = 0.0002
                twist.angular.z = -gain * self.green_error
                print(f"[GREEN] Erreur visuelle : {self.green_error} | Z : {twist.angular.z:.3f}")
            else:
                twist.linear.x = 0.02
                twist.angular.z = 0.0
                print("[GREEN] Ligne verte non détectée, avance lente...")

        elif self.state == 'stop':
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            print("⛔ Robot arrêté.")

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorFollower()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()