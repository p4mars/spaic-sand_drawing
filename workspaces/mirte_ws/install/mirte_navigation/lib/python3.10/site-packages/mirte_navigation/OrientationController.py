import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

import json

class OrientationController(Node):

    def __init__(self):
        super().__init__("orientation_controller")

        # ── Tuning parameters ────────────────────────────────────────────────
        self.angle_accuracy = 0.03
        self.base_kp        = 3.0
        self.base_max_vel   = 0.8

        # ── State ────────────────────────────────────────────────────────────
        self.frame = 0

        # ── Publishers ───────────────────────────────────────────────────────
        self.base_pub = self.create_publisher(
            Twist,
            "/mirte_base_controller/cmd_vel",
            10,
        )

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(String, "/target_angle", self._angle_callback, 10)

    def _angle_callback(self, msg: String) -> None:
        self.frame += 1
        data    = json.loads(msg.data)
        angle_x: float = data["angle_x"]

        x_active = abs(angle_x) > self.angle_accuracy

        # ── Base yaw correction ──────────────────────────────────────────────
        if x_active:
            yaw_rate = max(-self.base_max_vel,
                           min(self.base_max_vel, self.base_kp * angle_x))
            cmd = Twist()
            cmd.angular.z = yaw_rate
            self.base_pub.publish(cmd)
        else:
            self.base_pub.publish(Twist())

        # ── Arm pitch correction (disabled) ──────────────────────────────────
        # if y_active:
        #     delta = self.arm_kp * angle_y
        #     delta = max(-self.arm_max_delta, min(self.arm_max_delta, delta))
        #     self.elbow_angle = max(self.arm_min_angle,
        #                            min(self.arm_max_angle, self.elbow_angle + delta))
        #     traj  = JointTrajectory()
        #     traj.joint_names = [
        #         "shoulder_pan_joint",
        #         "shoulder_lift_joint",
        #         "elbow_joint",
        #         "wrist_joint",
        #     ]
        #     point = JointTrajectoryPoint()
        #     point.positions = [0.0, 0.0, self.elbow_angle, 0.0]
        #     traj.points.append(point)
        #     self.arm_pub.publish(traj)

        self.get_logger().info(
            f"Frame {self.frame}: "
            f"ax={angle_x:+.3f}  "
            f"base={'ON ' if x_active else 'off'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = OrientationController()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()