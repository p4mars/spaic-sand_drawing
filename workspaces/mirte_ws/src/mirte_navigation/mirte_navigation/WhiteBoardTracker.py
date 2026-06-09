import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


FRONT_CONE_HALF_ANGLE: float = 0.15
LIDAR_MIN_RANGE:       float = 0.1

ACTIVE_STATES: frozenset[str] = frozenset({"TRACK_WHITEBOARD", "TRACK_SANDPIT"})


class WhiteBoardTracker(Node):

    def __init__(self) -> None:
        super().__init__("vision_orientation_controller")

        # ── Tuning: searching ────────────────────────────────────────────────
        self.stop_distance:      float = 0.6
        self.target_timeout:     float = 5.0
        self.search_nudge_angle: float = 0.3
        self.search_nudge_speed: float = 0.3

        # ── Tuning: orientation correction ───────────────────────────────────
        self.angle_accuracy:         float = 0.05   # rad – deadband for "centred"
        self.centering_kp:           float = 3.0    # gentler gain while centering
        self.centering_max_vel:      float = 0.3
        self.centering_hold_time:    float = 0.5    # s centred before DONE fires
        self.base_kp:                float = 5.0
        self.base_max_vel:           float = 0.5
        self.track_linear:           float = 0.2

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
        self._centred_since:      float | None = None   # timestamp when centering lock began

        self.get_logger().info("Mode → SEARCHING  (wait-and-nudge sweep)")

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub   = self.create_publisher(Twist,  "/mirte_base_controller/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/state_change",                  10)

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

        if self.robot_state not in ACTIVE_STATES:
            return

        data = json.loads(msg.data)
        angle_x: float = data["angle_x"]
        self.last_angle = angle_x

        if self.mode == "DONE":
            return

        # ── Enter CENTERING once close enough ────────────────────────────────
        if self.front_distance < self.stop_distance and self.mode != "CENTERING":
            self._set_mode("CENTERING", f"distance {self.front_distance:.2f} m < {self.stop_distance} m")
            self._centred_since = None

        if self.mode == "CENTERING":
            self._apply_centering(angle_x)
            return

        # ── Normal TRACKING approach ─────────────────────────────────────────
        self._set_mode("TRACKING", "target detected")
        self._apply_orientation_correction(angle_x)

    def _scan_callback(self, msg: LaserScan) -> None:
        readings: list[float] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if (-1.57 - FRONT_CONE_HALF_ANGLE < angle < -1.57 + FRONT_CONE_HALF_ANGLE
                    and math.isfinite(r) and r > LIDAR_MIN_RANGE):
                readings.append(r)
            angle += msg.angle_increment
        self.front_distance = min(readings) if readings else float("inf")

    def _state_callback(self, msg: String) -> None:
        new_state = msg.data
        if new_state != self.robot_state and new_state in ACTIVE_STATES:
            self.get_logger().info(
                f"Robot state → {new_state}, resetting tracker to SEARCHING"
            )
            self.mode = "SEARCHING"
            self.last_detection_time = time.time()
            self._nudging = False
            self._nudge_remaining = 0.0
            self._centred_since = None
        self.robot_state = new_state

    # -- Orientation correction ----------------------------------------------

    def _apply_orientation_correction(self, angle_x: float) -> None:
        """Normal proportional approach used during TRACKING."""
        x_active = abs(angle_x) > self.angle_accuracy

        cmd = Twist()
        if x_active:
            cmd.angular.z = self._clamp(self.base_kp * angle_x, self.base_max_vel)
        else:
            cmd.linear.x = self.track_linear
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Frame {self.frame}: "
            f"ax={angle_x:+.3f}  "
            f"base={'ON' if x_active else 'off'}  "
            f"fwd={'ON' if not x_active else 'off'}"
        )

    def _apply_centering(self, angle_x: float) -> None:
        """
        Rotate in place to bring the target to the image centre.
        Once centred for ``centering_hold_time`` seconds, stop and publish DONE.
        """
        centred = abs(angle_x) < self.angle_accuracy

        if centred:
            if self._centred_since is None:
                self._centred_since = time.time()
                self.get_logger().info("Centred – holding to confirm…")

            if time.time() - self._centred_since >= self.centering_hold_time:
                # Confirmed centred – we're done
                self.cmd_pub.publish(Twist())
                self._set_mode("DONE", "centred on target")
                done_msg = String()
                done_msg.data = "DONE"
                self.state_pub.publish(done_msg)
                return
        else:
            # Lost centre lock – reset hold timer
            self._centred_since = None

        # Rotate gently toward centre; no forward motion
        cmd = Twist()
        cmd.angular.z = self._clamp(self.centering_kp * angle_x, self.centering_max_vel)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"CENTERING frame {self.frame}: ax={angle_x:+.3f}  "
            f"{'LOCKED' if centred else 'rotating'}"
        )

    # -- Background tick -----------------------------------------------------

    def _background_tick(self) -> None:
        if self.robot_state not in ACTIVE_STATES:
            return
        if self.mode in ("TRACKING", "DONE", "CENTERING"):
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
    node = WhiteBoardTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()