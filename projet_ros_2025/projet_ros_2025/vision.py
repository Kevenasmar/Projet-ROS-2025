import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
import numpy as np
import cv2

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is not None:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Yellow mask
            lower_yellow = np.array([20, 80, 50])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # Red mask
            lower_red = np.array([171, 100, 120])
            upper_red = np.array([175, 255, 137])
            mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

            # Combine masks
            combined_mask = cv2.bitwise_or(mask_yellow, mask_red)
            result = cv2.bitwise_and(image, image, mask=combined_mask)

            # --- Detect individual red goal posts ---
            contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            centres = []

            for i, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centres.append((cx, cy))
                    cv2.circle(result, (cx, cy), 5, (255, 0, 0), -1)
                    
                    # Texte avec numéro + coordonnées
                    label = f"Poteau {i+1}: ({cx},{cy})"
                    cv2.putText(result, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)


            # --- Centre du but ---
            if len(centres) == 2:
                cx_mid = (centres[0][0] + centres[1][0]) // 2
                cy_mid = (centres[0][1] + centres[1][1]) // 2
                cv2.circle(result, (cx_mid, cy_mid), 6, (0, 255, 0), -1)
                cv2.putText(result, f"({cx_mid},{cy_mid})", (cx_mid + 10, cy_mid + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # --- Detect yellow ball with only the center of the largest circle ---
            twist = Twist()
            yellow_blur = cv2.GaussianBlur(mask_yellow, (9, 9), 2)
            circles = cv2.HoughCircles(
                yellow_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30, minRadius=5, maxRadius=100
            )

            if circles is not None and len(circles[0]) > 0:
                largest_circle = max(np.uint16(np.around(circles))[0], key=lambda c: c[2])
                cx_ball, cy_ball, radius = largest_circle

                # Afficher uniquement le centre
                cv2.circle(result, (cx_ball, cy_ball), 5, (0, 0, 255), -1)
                cv2.putText(result, f"Balle: ({cx_ball},{cy_ball})", (cx_ball + 10, cy_ball),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
            cv2.line(result, (0, 240), (result.shape[1], 240), (0, 0, 255), 2)  # (0, 0, 255) is red in BGR

            for y in [180, 300]:
                for x in range(0, result.shape[1], 10):  # Step of 10 pixels for dashed effect
                    cv2.line(result, (x, y), (x + 5, y), (0, 0, 255), 2)

            # Display result
            cv2.imshow("Yellow + Red Detection + Centre du But", result)
            cv2.waitKey(1)

        else:
            self.get_logger().warn("Failed to decode compressed image")

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
