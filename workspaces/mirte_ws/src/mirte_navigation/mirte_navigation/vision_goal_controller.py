import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion


class VisionGoalController(Node):

    def __init__(self):

        super().__init__('vision_goal_controller')

        # Publisher
        self.cmd_pub = self.create_publisher(
            Twist,
            '/mirte_base_controller/cmd_vel_unstamped',
            10
        )

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10
        )

        # Pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Scan
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')

        # State machine
        self.state = 'driving'
        self.stage = "navigate"
        self.turn_start = None

        # Waypoints (RELATIVE to start pose)
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = 0.0
        self.goal_received = False

        # start pose
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.goal_initialized = False
        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

    # =========================
    # SCAN
    # =========================

    def scan_callback(self, msg):

        angle = msg.angle_min

        front = []
        left = []
        right = []

        for r in msg.ranges:

            if math.isfinite(r) and r > 0.15:

                if -0.35 < angle < 0.35:
                    front.append(r)

                elif 0.35 <= angle < 1.2:
                    left.append(r)

                elif -1.2 < angle <= -0.35:
                    right.append(r)

            angle += msg.angle_increment

        self.front_distance = min(front) if front else float('inf')
        self.left_distance = min(left) if left else float('inf')
        self.right_distance = min(right) if right else float('inf')

    # =========================
    # ODOM
    # =========================

    def odom_callback(self, msg):

        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        (_, _, self.yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        if not self.goal_initialized:
            self.start_x = self.x
            self.start_y = self.y
            self.start_yaw = self.yaw
            self.goal_initialized = True

    # =========================
    # GOAL
    # =========================

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

        # yaw from quaternion (optional but correct)
        q = msg.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.goal_yaw = yaw
        self.goal_received = True

        self.get_logger().info(
            f"New goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}, yaw={self.goal_yaw:.2f}"
        )

    # =========================
    # TRANSFORM RELATIVE → WORLD
    # =========================

    def get_world_goal(self, gx, gy, gyaw):

        x = self.start_x + (gx * math.cos(self.start_yaw) - gy * math.sin(self.start_yaw))
        y = self.start_y + (gx * math.sin(self.start_yaw) + gy * math.cos(self.start_yaw))
        yaw = self.start_yaw + gyaw

        return x, y, yaw

    # =========================
    # CONTROL LOOP
    # =========================

    def control_loop(self):
        if not self.goal_received:
            return
        if self.stage == "done":
            self.cmd_pub.publish(Twist())
            return
        if not self.goal_initialized:
            return

        msg = Twist()

        STOP_DIST = 0.4
        CLEAR_DIST = 0.6

        # =========================
        # OBSTACLE PRIORITY
        # =========================

        if self.state == 'driving':

            if self.front_distance < STOP_DIST:
                self.state = 'turning'
                self.turn_start = time.time()

        if self.state == 'turning':

            if self.left_distance > self.right_distance:
                msg.angular.z = 0.5
            else:
                msg.angular.z = -0.5

            msg.linear.x = 0.0
            self.cmd_pub.publish(msg)

            if self.front_distance > CLEAR_DIST:
                self.turn_start = time.time()
                self.state = 'driving'

            return

        # =========================
        # CURRENT GOAL
        # =========================

        goal_x, goal_y, goal_yaw = self.get_world_goal(
            self.goal_x,
            self.goal_y,
            self.goal_yaw
        )

        dx = goal_x - self.x
        dy = goal_y - self.y

        distance = math.sqrt(dx**2 + dy**2)

        # =========================
        # ARRIVED AT POSITION
        # =========================

        if distance < 0.05 and self.stage == "navigate":
            self.stage = "rotate_final"
            return

        # =========================
        # FINAL ORIENTATION
        # =========================

        if self.stage == "rotate_final":

            angle_error = math.atan2(
                math.sin(self.goal_yaw - self.yaw),
                math.cos(self.goal_yaw - self.yaw)
            )

            if abs(angle_error) > 0.05:
                msg.angular.z = 1.5 * angle_error
            else:
                msg.angular.z = 0.0
                self.get_logger().info(f"Waypoint complete")
                self.stage = "done"
                self.cmd_pub.publish(Twist())
                self.get_logger().info("MISSION COMPLETE")
                return
            self.cmd_pub.publish(msg)
            return

        # =========================
        # NAVIGATION
        # =========================

        target_angle = math.atan2(dy, dx)

        angle_error = math.atan2(
            math.sin(target_angle - self.yaw),
            math.cos(target_angle - self.yaw)
        )
        if self.turn_start is None:
            if abs(angle_error) > 0.1:
                msg.angular.z = max(min(1.5 * angle_error, 1.0), -1.0)
            else:
                msg.linear.x = min(0.3, 0.5 * distance)
        else :
            if abs(angle_error) > 0.1 and time.time() - self.turn_start > 3.0:
                msg.angular.z = max(min(1.5 * angle_error, 1.0), -1.0)
            else:
                msg.linear.x = min(0.3, 0.5 * distance)

        self.cmd_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = VisionGoalController()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()