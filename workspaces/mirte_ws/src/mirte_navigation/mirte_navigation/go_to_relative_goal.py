import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf_transformations import euler_from_quaternion


class GoToGoal(Node):

    def __init__(self, yaw_input):

        super().__init__('go_to_goal')

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
        self.waypoints = []
        self.current_goal = 0

        # start pose
        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.goal_initialized = False

        # yaw mode
        self.yaw_input = yaw_input

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # =========================
        # INPUT WAYPOINTS
        # =========================

        print("Enter waypoints (x y yaw_deg). Type 'done' when finished")

        while True:
            x = input("X (or done): ")

            if x.lower() == "done":
                break

            y = float(input("Y: "))
            yaw = float(input("Yaw (deg): "))

            self.waypoints.append((float(x), y, math.radians(yaw)))

        if len(self.waypoints) == 0:
            raise Exception("No waypoints provided")

        self.get_logger().info(f"{len(self.waypoints)} waypoints loaded")

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
        if self.stage == "done":
            self.cmd_pub.publish(Twist())
            return

        if self.current_goal >= len(self.waypoints):
            self.stage = "done"
            self.cmd_pub.publish(Twist())
            return

        if not self.goal_initialized:
            return

        msg = Twist()

        STOP_DIST = 0.4
        CLEAR_DIST = 0.5

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
        # CURRENT WAYPOINT
        # =========================

        gx, gy, gyaw = self.waypoints[self.current_goal]
        goal_x, goal_y, goal_yaw = self.get_world_goal(gx, gy, gyaw)

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
                math.sin(goal_yaw - self.yaw),
                math.cos(goal_yaw - self.yaw)
            )

            if abs(angle_error) > 0.05:
                msg.angular.z = 1.5 * angle_error
            else:
                msg.angular.z = 0.0
                self.get_logger().info(f"Waypoint {self.current_goal + 1} complete")

                self.current_goal += 1

                if self.current_goal >= len(self.waypoints):
                    self.stage = "done"
                    self.cmd_pub.publish(Twist())
                    self.get_logger().info("MISSION COMPLETE")
                    return

                input("Press ENTER for next waypoint...")

                self.stage = "navigate"

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

    yaw_input = input("Default final yaw (deg, optional, ENTER for none): ")

    node = GoToGoal(yaw_input)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()