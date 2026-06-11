import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


FRONT_CONE_HALF_ANGLE: float = 0.15
LIDAR_MIN_RANGE:       float = 0.1
SCAN_OFFSET = -1.57
ACTIVE_STATES: frozenset[str] = frozenset({"TRACK_WHITEBOARD", "TRACK_SANDPIT"})

# ── Per-state tuning profiles ─────────────────────────────────────────────────
PROFILES: dict[str, dict] = {
    "TRACK_WHITEBOARD": {
        "stop_distance":       1.5,
        "target_timeout":      5.0,
        "search_nudge_angle":  0.5,
        "search_nudge_speed":  0.3,
        "angle_accuracy":      0.1,
        "base_kp":             4.0,
        "base_max_vel":        0.5,
        "track_linear":        0.3,
        "centering_kp":        2.5,
        "centering_max_vel":   0.3,
        "centering_hold_time": 1.0,
    },
    "TRACK_SANDPIT": {
        "stop_distance":       0.5,
        "target_timeout":      5.0,
        "search_nudge_angle":  0.5,
        "search_nudge_speed":  0.2,
        "angle_accuracy":      0.1,
        "base_kp":             5.0,
        "base_max_vel":        0.6,
        "track_linear":        0.2,
        "centering_kp":        6.0,
        "centering_max_vel":   0.25,
        "centering_hold_time": 0.5,
    },
}


class WhiteBoardTracker(Node):

    def __init__(self) -> None:
        super().__init__("vision_orientation_controller")

        # ── Active tuning (loaded from PROFILES on state change) ─────────────
        self._load_profile("TRACK_WHITEBOARD")

        # ── State ────────────────────────────────────────────────────────────
        self.mode:                str          = "SEARCHING"
        self.robot_state:         str          = "IDLE"
        self.frame:               int          = 0
        self.last_angle:          float        = 0.0
        self.front_distance:      float        = float("inf")
        self.recovery_start:      float | None = None
        self._nudging:            bool         = False
        self._nudge_remaining:    float        = 0.0
        self._centred_since:      float | None = None
        self.last_detection_time: float | None = None  # None until first detection

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

    # -- Profile loading ------------------------------------------------------

    def _load_profile(self, state: str) -> None:
        p = PROFILES[state]
        self.stop_distance       = p["stop_distance"]
        self.target_timeout      = p["target_timeout"]
        self.search_nudge_angle  = p["search_nudge_angle"]
        self.search_nudge_speed  = p["search_nudge_speed"]
        self.angle_accuracy      = p["angle_accuracy"]
        self.base_kp             = p["base_kp"]
        self.base_max_vel        = p["base_max_vel"]
        self.track_linear        = p["track_linear"]
        self.centering_kp        = p["centering_kp"]
        self.centering_max_vel   = p["centering_max_vel"]
        self.centering_hold_time = p["centering_hold_time"]
        self.get_logger().info(
            f"Loaded profile '{state}': "
            f"stop_distance={self.stop_distance} m  "
            f"track_linear={self.track_linear} m/s"
        )

    # -- Helpers --------------------------------------------------------------

    def _set_mode(self, new_mode: str, extra: str = "") -> None:
        if new_mode == self.mode:
            return
        suffix = f"  ({extra})" if extra else ""
        self.get_logger().info(f"Mode  {self.mode}  →  {new_mode}{suffix}")
        self.mode = new_mode

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    # -- Callbacks ------------------------------------------------------------

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
            return

        if self.mode == "CENTERING":
            self._apply_centering(angle_x)
            return

        # ── Normal TRACKING approach ──────────────────────────────────────────
        self._set_mode("TRACKING", "target detected")
        self._apply_orientation_correction(angle_x)

    def _scan_callback(self, msg: LaserScan) -> None:
        readings: list[float] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if (SCAN_OFFSET - FRONT_CONE_HALF_ANGLE < angle < SCAN_OFFSET + FRONT_CONE_HALF_ANGLE
                    and math.isfinite(r) and r > LIDAR_MIN_RANGE):
                readings.append(r)
            angle += msg.angle_increment
        self.front_distance = min(readings) if readings else float("inf")

    def _state_callback(self, msg: String) -> None:
        new_state = msg.data
        if new_state != self.robot_state and new_state in ACTIVE_STATES:
            self.get_logger().info(f"Robot state → {new_state}, resetting tracker")
            self._load_profile(new_state)
            self.mode = "SEARCHING"
            self.last_detection_time = time.time()
            self._nudging = False
            self._nudge_remaining = 0.0
            self._centred_since = None
        self.robot_state = new_state

    # -- Orientation correction -----------------------------------------------

    def _apply_orientation_correction(self, angle_x: float) -> None:
        x_active = abs(angle_x) > self.angle_accuracy
        cmd = Twist()
        if x_active:
            cmd.angular.z = self._clamp(self.base_kp * angle_x, self.base_max_vel)
        else:
            cmd.linear.x = self.track_linear
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Frame {self.frame}: ax={angle_x:+.3f}  "
            f"base={'ON' if x_active else 'off'}  "
            f"fwd={'ON' if not x_active else 'off'}"
        )

    def _apply_centering(self, angle_x: float) -> None:
        centred = abs(angle_x) < self.angle_accuracy

        if centred:
            if self._centred_since is None:
                self._centred_since = time.time()
                self.get_logger().info("Centred – holding to confirm…")
            if time.time() - self._centred_since >= self.centering_hold_time:
                self.cmd_pub.publish(Twist())
                self._set_mode("DONE", "centred on target")
                done_msg = String()
                done_msg.data = "DONE"
                self.state_pub.publish(done_msg)
                return
        else:
            self._centred_since = None

        cmd = Twist()
        cmd.angular.z = self._clamp(self.centering_kp * angle_x, self.centering_max_vel)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"CENTERING frame {self.frame}: ax={angle_x:+.3f}  "
            f"{'LOCKED' if centred else 'rotating'}"
        )

    # -- Background tick ------------------------------------------------------

    def _background_tick(self) -> None:
        if self.robot_state not in ACTIVE_STATES:
            return
        if self.mode in ("DONE", "CENTERING"):
            return
        if self.last_detection_time is None:
            return  # detector not ready yet, don't nudge or recover
        cmd = Twist()
        time_since_seen = time.time() - self.last_detection_time

        if self.mode == "RECOVERY":
            elapsed = time.time() - self.recovery_start
            if elapsed < 5.0:
                cmd.linear.x = -0.1
            elif elapsed < 3.0:
                cmd.angular.z = -0.4 if self.last_angle > 0 else -0.4
            else:
                self.last_detection_time = time.time()
                self._nudging = False
                self._set_mode("SEARCHING", "recovery done")
            self.cmd_pub.publish(cmd)
            return

        if self.mode == "TRACKING" and time_since_seen > self.target_timeout:
            # ── If already within stop distance, marker likely left FOV due to
            # proximity – jump to CENTERING rather than backing away
            if self.front_distance < self.stop_distance:
                self.get_logger().info(
                    f"Detection lost but within stop distance "
                    f"({self.front_distance:.2f} m < {self.stop_distance} m) – "
                    f"switching to CENTERING on last known angle"
                )
                self._set_mode("DONE")
                cmd = Twist()
                cmd.angular.z = self._clamp(self.centering_kp * angle_x, self.centering_max_vel)
                self.cmd_pub.publish(cmd)
                self._centred_since = None
                return

            self._set_mode("RECOVERY", "target lost")
            self.recovery_start = time.time()
            return

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