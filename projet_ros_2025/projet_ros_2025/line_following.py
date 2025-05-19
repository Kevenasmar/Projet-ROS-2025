import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        self.declare_parameter('roundabout_direction', 'right') 
        self.roundabout_direction = self.get_parameter('roundabout_direction').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        self.blue_handled = False
        self.emergency_stop = False  # Flag d'arrêt d'urgence
        self.get_logger().info("Node démarré. Direction du rond-point: " + self.roundabout_direction)
    
    def laser_callback(self, scan_msg):
        np_array = np.array(scan_msg.ranges)
        front = np.mean(np.concatenate((np_array[-10:], np_array[:10])))
        left = np.mean(np_array[80:100])
        back = np.mean(np_array[170:190])
        right = np.mean(np_array[260:280])
        distances = [front, left, back, right]
        if min(distances) < 0.3:
            self.emergency_stop = True
            self.get_logger().warn("⚠️ Obstacle détecté : arrêt d'urgence !")
        else:
            self.emergency_stop = False
        
    def image_callback(self, msg):

        # Convertir les données compressées en tableau numpy
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Décoder l'image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        height, width, _ = frame.shape
        roi = hsv[int(height * 0.6):, :]

        twist = Twist()
        
        # Détection du bleu
        mask_blue = cv2.inRange(roi, (85, 50, 50), (130, 255, 255))
        M_blue = cv2.moments(mask_blue)
        blue_area = M_blue['m00']

        # Détection du rouge
        mask_red1 = cv2.inRange(roi, (0, 70, 50), (10, 255, 255))
        mask_red2 = cv2.inRange(roi, (160, 70, 50), (180, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Détection du vert
        mask_green = cv2.inRange(roi, (30, 40, 40), (100, 255, 255))

        M_red = cv2.moments(mask_red)
        cx_red = int(M_red['m10'] / M_red['m00']) if M_red['m00'] > 0 else None

        M_green = cv2.moments(mask_green)
        cx_green = int(M_green['m10'] / M_green['m00']) if M_green['m00'] > 0 else None

        if cx_red is not None and cx_green is not None:
            if cx_green < cx_red:
                cx = int((cx_red + cx_green) / 2)
                err = cx - (width // 2)
                twist.linear.x = 0.12
                twist.angular.z = -float(err) / 200.0
            else:
                self.get_logger().warn("Lignes inversées (vert à droite de rouge)")
                twist.angular.z = 0.2 if self.roundabout_direction == "left" else -0.2
        elif cx_red is not None and cx_red < 550:
            self.get_logger().warn("Seulement la ligne rouge détectée – tournant à gauche")
            twist.linear.x = 0.06
            twist.angular.z = 0.50
        elif cx_green is not None and cx_green < 550:
            self.get_logger().warn("Seulement la ligne verte détectée – tournant à droite")
            twist.linear.x = 0.06
            twist.angular.z = -0.50
        else:
            self.get_logger().warn("Aucune ligne détectée – continue tout droit doucement")
            twist.linear.x = 0.06
            twist.angular.z = 0.0

        if not self.emergency_stop:
            self.cmd_pub.publish(twist)
        else:
            stop_msg = Twist()
            self.cmd_pub.publish(stop_msg)

        if blue_area > 2000000:
            left_half = mask_blue[:, :width // 2]
            right_half = mask_blue[:, width // 2:]

            left_area = cv2.moments(left_half)['m00']
            right_area = cv2.moments(right_half)['m00']

            if abs(left_area - right_area) < 500000:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.blue_handled = True
                self.get_logger().info("Alignement réussi – arrêt du robot et fin du programme.")
                rclpy.shutdown()
                return
            elif left_area > right_area:
                twist.linear.x = 0.0
                twist.angular.z = 0.45
            else:
                twist.linear.x = 0.0
                twist.angular.z = -0.45

            self.cmd_pub.publish(twist)
            return


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()
    cv2.destroyAllWindows() 