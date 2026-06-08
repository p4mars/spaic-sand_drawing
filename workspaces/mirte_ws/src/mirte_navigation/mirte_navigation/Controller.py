import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


FRONT_CONE_HALF_ANGLE: float = 0.3
LIDAR_MIN_RANGE:       float = 0.1


class VisionOrientationController(Node):

    def __init__(self) -> None:
        super().__init__("vision_orientation_controller")

        # ── Tuning: searching ────────────────────────────────────────────────
        self.stop_distance:      float = 2.5
        self.target_timeout:     float = 5.0
        self.search_nudge_angle: float = 0.3
        self.search_nudge_speed: float = 0.3

        # ── Tuning: orientation correction ───────────────────────────────────
        self.angle_accuracy: float = 0.1
        self.base_kp:        float = 5.0
        self.base_max_vel:   float = 0.8
        self.track_linear:   float = 0.5

        # ── State ────────────────────────────────────────────────────────────
        self.mode:                str         = "SEARCHING"
        self.robot_state:         str         = "IDLE"
        self.frame:               int         = 0
        self.last_angle:          float       = 0.0
        self.front_distance:      float       = float("inf")
        self.last_detection_time: float       = time.time()
        self.recovery_start:      float | None = None
        self._nudging:            bool        = False
        self._nudge_remaining:    float       = 0.0

        self.get_logger().info("Mode → SEARCHING  (wait-and-nudge sweep)")

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub   = self.create_publisher(Twist,          "/mirte_base_controller/cmd_vel",                10)
        self.state_pub = self.create_publisher(String,         "/state_change",                                 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(String,    "/target_angle", self._angle_callback, 10)
        self.create_subscription(LaserScan, "/scan",         self._scan_callback,  10)
        self.create_subscription(String,    "/robot_state",  self._state_callback, 10)

        # ── Background tick (10 Hz) ───────────────────────────────────────────
        self.create_timer(0.1, self._background_tick)

    # -- Helpers -------------------------------------------------------------

    def _set_mode(self, new_mode: str, extra: str = "") -> None:
        if new_mode == self.mode:
            return
        suffix = f"  ({extra})" if extra else ""
        self.get_logger().info(f"Mode  {self.mode}  →  {new_mode}{suffix}")
        self.mode = new_mode

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    # -- Callbacks -----------------------------------------------------------

    def _angle_callback(self, msg: String) -> None:
        self.last_detection_time = time.time()
        self._nudging = False
        self.frame += 1

        if self.robot_state != "TRACK_WHITEBOARD":
            return

        data    = json.loads(msg.data)
        angle_x: float = data["angle_x"]
        self.last_angle = angle_x

        if self.mode != "DONE":
            self._set_mode("TRACKING", "target detected")

        if self.mode == "DONE":
            return

        # ── TRACKING → DONE when close enough ────────────────────────────────
        if self.front_distance < self.stop_distance:
            self._set_mode("DONE", f"distance {self.front_distance:.2f} m < {self.stop_distance} m")
            self.cmd_pub.publish(Twist())
            done_msg = String()
            done_msg.data = "DONE"
            self.state_pub.publish(done_msg)
            return

        self._apply_orientation_correction(angle_x)

    def _scan_callback(self, msg: LaserScan) -> None:
        readings: list[float] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if (-FRONT_CONE_HALF_ANGLE < angle < FRONT_CONE_HALF_ANGLE
                    and math.isfinite(r) and r > LIDAR_MIN_RANGE):
                readings.append(r)
            angle += msg.angle_increment
        self.front_distance = min(readings) if readings else float("inf")

    def _state_callback(self, msg: String) -> None:
        self.robot_state = msg.data

    # -- Orientation correction (x-axis only) --------------------------------

    def _apply_orientation_correction(self, angle_x: float) -> None:
        x_active = abs(angle_x) > self.angle_accuracy

        if x_active:
            cmd = Twist()
            cmd.angular.z = self._clamp(self.base_kp * angle_x, self.base_max_vel)
            self.cmd_pub.publish(cmd)
        else:
            # Centred → drive forward
            cmd = Twist()
            cmd.linear.x = self.track_linear
            self.cmd_pub.publish(cmd)

        # ── Arm pitch correction (disabled) ──────────────────────────────────
        # if y_active:
        #     delta = self._clamp(self.arm_kp * angle_y, self.arm_max_delta)
        #     self.elbow_angle = max(self.arm_min_angle,
        #                            min(self.arm_max_angle, self.elbow_angle + delta))
        #     traj = JointTrajectory()
        #     traj.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
        #                          "elbow_joint", "wrist_joint"]
        #     point = JointTrajectoryPoint()
        #     point.positions = [0.0, 0.0, self.elbow_angle, 0.0]
        #     traj.points.append(point)
        #     self.arm_pub.publish(traj)

        self.get_logger().info(
            f"Frame {self.frame}: "
            f"ax={angle_x:+.3f}  "
            f"base={'ON' if x_active else 'off'}  "
            f"fwd={'ON' if not x_active else 'off'}"
        )

    # -- Background tick -----------------------------------------------------

    def _background_tick(self) -> None:
        if self.robot_state != "TRACK_WHITEBOARD":
            return
        if self.mode in ("TRACKING", "DONE"):
            return

        cmd = Twist()
        time_since_seen = time.time() - self.last_detection_time

        # ── RECOVERY ─────────────────────────────────────────────────────────
        if self.mode == "RECOVERY":
            elapsed = time.time() - self.recovery_start
            if elapsed < 1.0:
                cmd.linear.x = -0.1
            elif elapsed < 3.0:
                cmd.angular.z = 0.4 if self.last_angle > 0 else -0.4
            else:
                self.last_detection_time = time.time()
                self._nudging = False
                self._set_mode("SEARCHING", "recovery done")
            self.cmd_pub.publish(cmd)
            return

        # ── TRACKING → RECOVERY on lost target ───────────────────────────────
        if self.mode == "TRACKING" and time_since_seen > self.target_timeout:
            self._set_mode("RECOVERY", "target lost")
            self.recovery_start = time.time()

        # ── SEARCHING: wait → nudge loop ─────────────────────────────────────
        if self.mode == "SEARCHING":
            if not self._nudging:
                if time_since_seen > self.target_timeout:
                    self._nudge_remaining = self.search_nudge_angle
                    self._nudging = True
                    self.get_logger().info(
                        f"No detection for {self.target_timeout:.0f} s – nudging "
                        f"{math.degrees(self.search_nudge_angle):.0f} °"
                    )
            else:
                step = self.search_nudge_speed * 0.1
                if self._nudge_remaining > 0:
                    cmd.angular.z = self.search_nudge_speed
                    self._nudge_remaining -= step
                else:
                    self._nudging = False
                    self.last_detection_time = time.time()
                    self.get_logger().info("Nudge complete – waiting for detection")

            self.cmd_pub.publish(cmd)


# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionOrientationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()