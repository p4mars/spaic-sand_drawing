"""
vision_controller.py
--------------------
ROS 2 node that drives the Mirte robot toward a visually detected target.

State machine
─────────────
  INIT_SCAN  →  SEARCHING  →  TRACKING  →  EXPLORE  →  DONE
                                  ↓          (lateral scan + fine-align)
                              RECOVERY
                                  ↓
                              SEARCHING

INIT_SCAN
  On start-up the robot performs one full in-place rotation (360 °) so that
  any target already in the scene is detected before the outward spiral begins.
  If a target is spotted during INIT_SCAN the node transitions immediately to
  TRACKING, just as it would in SEARCHING.

EXPLORE sub-states (executed in order)
  LEFT         – strafe left for one step
  LEFT_ALIGN   – stop and rotate until camera is centred on target
  RIGHT        – strafe right two steps (left → centre → right)
  RIGHT_ALIGN  – stop and rotate until camera is centred on target
  DECIDE       – compare bbox areas; choose best lateral position
  RETURN_*     – strafe back to the chosen position
  FINAL_ALIGN  – rotate until camera is centred; then → DONE

Subscriptions
  /target_angle  (std_msgs/String)    – JSON {"angle": float, "bbox_area": float}
  /scan          (sensor_msgs/LaserScan) – LiDAR data

Publications
  /mirte_base_controller/cmd_vel_unstamped     (geometry_msgs/Twist)
  /mirte_master_arm_controller/joint_trajectory (trajectory_msgs/JointTrajectory)
"""

import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LiDAR cone half-angle (rad) used to extract "forward" readings
FRONT_CONE_HALF_ANGLE: float = 0.3

# Minimum plausible LiDAR return (m) – filters noise / self-hits
LIDAR_MIN_RANGE: float = 0.1

# Proportional gain: angle error (rad) → angular velocity (rad/s)
ANGLE_KP: float = 0.4

# Hard cap on commanded angular velocity (rad/s)
MAX_ANGULAR: float = 0.2

# Forward cruise speed while tracking (m/s)
TRACK_LINEAR: float = 0.4

# Lateral (y) speed during EXPLORE crab-walk steps (m/s)
CRAB_SPEED: float = 0.05

# Duration of a single EXPLORE lateral step (s); two steps = full sweep
CRAB_STEP_DURATION: float = 2.0

# Angle tolerance used to decide "camera is centred" during EXPLORE (rad)
ALIGN_TOLERANCE: float = math.radians(1)

# Angular speed used during the initial 360 ° scan (rad/s).
# A full revolution takes  2π / INIT_SCAN_ANGULAR  seconds.
INIT_SCAN_ANGULAR: float = 0.1
INIT_SCAN_DURATION: float = 2 * math.pi / INIT_SCAN_ANGULAR 

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class VisionController(Node):
    """
    Visual-servoing controller for the Mirte mobile manipulator.

    The robot first rotates a full 360 ° in place (INIT_SCAN) to check for
    any target already visible before beginning the outward search spiral.
    Once a target is found it drives toward it, then performs a short lateral
    scan (left / right) to find the viewpoint with the largest bounding box.
    The camera is re-centred between lateral steps so that bbox-area readings
    are taken from a consistent heading.
    """

    # -- Lifecycle -----------------------------------------------------------

    def __init__(self) -> None:
        super().__init__("vision_controller")

        # ── Publishers ──────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            "/mirte_base_controller/cmd_vel_unstamped",
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            '/state_change',
            10
        )

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(
            String,
            "/target_angle",
            self._angle_callback,
            10,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            String,
            "/robot_state",
            self.state_callback,
            10
        )

        # ── Sensor / detection state ─────────────────────────────────────────
        self.target_angle: float | None = None    # latest angle from vision (rad)
        self.last_angle: float = 0.0              # angle at last detection (recovery)
        self.front_distance: float = float("inf") # nearest forward obstacle (m)
        self.last_detection_time: float = 0.0     # epoch of last /target_angle msg
        self.target_size: float = 0.0             # bbox area from latest detection

        # ── Tunable parameters ───────────────────────────────────────────────
        # Distance at which the robot considers the target "reached" (m)
        self.stop_distance: float = 1.5

        # Angle error below which the robot drives forward instead of rotating
        self.angle_threshold: float = math.radians(5)

        # Seconds without a detection before TRACKING → RECOVERY
        self.target_timeout: float = 5.0

        # SEARCHING: constant forward speed (m/s)
        self.search_linear: float = 0.1

        # SEARCHING: angular speed starts high and decays (spiral outward)
        self.search_angular_start: float = 0.5  # initial  (rad/s)
        self.search_angular_min: float = 0.1    # floor    (rad/s)
        self.search_expand_rate: float = 0.05   # decay    (rad/s per second)

        # ── Finite-state machine ─────────────────────────────────────────────
        # Start with the initial 360 ° scan before any forward searching.
        self.mode: str = "INIT_SCAN"
        self.init_scan_start: float = time.time()  # epoch when INIT_SCAN began
        self.search_start: float = 0.0             # epoch when SEARCHING began
        self.recovery_start: float | None = None   # epoch when RECOVERY began
        self.state = "IDLE"
        self.get_logger().info("State → INIT_SCAN  (performing 360 ° sweep)")

        # ── EXPLORE sub-state ────────────────────────────────────────────────
        self.explore_mode: str | None = None  # current EXPLORE sub-state
        self.explore_start: float = 0.0       # epoch of current sub-state start
        self.best_size: float = 0.0           # bbox area at the starting position
        self.left_size: float = 0.0           # bbox area measured at left step
        self.right_size: float = 0.0          # bbox area measured at right step

        # ── Main control loop (10 Hz) ────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)

    # -- Mode helper ---------------------------------------------------------

    def _set_mode(self, new_mode: str, extra: str = "") -> None:
        """
        Transition to *new_mode*, logging the change.

        Parameters
        ----------
        new_mode:
            Target state name (e.g. ``"TRACKING"``).
        extra:
            Optional context appended to the log message
            (e.g. a distance reading or sub-state name).
        """
        if new_mode == self.mode:
            return  # no-op – already in that state

        suffix = f"  ({extra})" if extra else ""
        self.get_logger().info(f"State  {self.mode}  →  {new_mode}{suffix}")
        self.mode = new_mode

    # -- Callbacks -----------------------------------------------------------

    def _angle_callback(self, msg: String) -> None:
        """
        Receive the latest detection from the vision pipeline.

        The message is a JSON string with fields:
          ``angle``     – horizontal angle to target centre (rad)
          ``bbox_area`` – bounding-box pixel area (proxy for proximity/angle)

        A positive angle means the target is to the left; negative to the right.
        """
        data = json.loads(msg.data)
        angle: float = data["angle"]

        self.target_angle = angle
        self.last_angle = angle           # remember for RECOVERY rotation direction
        self.last_detection_time = time.time()
        self.target_size = data["bbox_area"]

        # Switch to TRACKING from search/scan/recovery states; never interrupt
        # EXPLORE or DONE with an incoming detection.
        if self.mode not in ("EXPLORE", "DONE"):
            self._set_mode("TRACKING", "target detected")

    def _scan_callback(self, msg: LaserScan) -> None:
        """
        Compute the minimum range inside the forward LiDAR cone and store it
        in ``self.front_distance``.
        """
        front_readings: list[float] = []
        angle = msg.angle_min

        for r in msg.ranges:
            in_front = -FRONT_CONE_HALF_ANGLE < angle < FRONT_CONE_HALF_ANGLE
            valid = math.isfinite(r) and r > LIDAR_MIN_RANGE

            if valid and in_front:
                front_readings.append(r)

            angle += msg.angle_increment

        self.front_distance = min(front_readings) if front_readings else float("inf")

    def state_callback(self,msg):
        self.state = msg.data
    
    # -- Control loop --------------------------------------------------------

    def _control_loop(self) -> None:
        """
        10 Hz state-machine tick.

        State priority (highest first):
          DONE      – task finished; robot stopped
          RECOVERY  – brief reverse then spin toward last known angle
          INIT_SCAN – one full 360 ° rotation before the outward spiral
          SEARCHING – forward spiral until a target appears
          EXPLORE   – lateral scan with camera realignment between steps
          TRACKING  – proportional angular + forward drive toward target
        """
        cmd = Twist()  # default: all zeros (stop)
        time_since_seen = time.time() - self.last_detection_time
        if self.state != "TRACK_WHITEBOARD":
            return

        # ── DONE ─────────────────────────────────────────────────────────────
        if self.mode == "DONE":
            self.cmd_pub.publish(Twist())  # ensure robot is stopped
            done = String()
            done.data = self.mode
            self.state_pub.publish(done)
            return

        # ── Transition: TRACKING → RECOVERY on detection timeout ─────────────
        if self.mode == "TRACKING" and time_since_seen > self.target_timeout:
            self._set_mode("RECOVERY", "target lost")
            self.recovery_start = time.time()

        # ── RECOVERY ─────────────────────────────────────────────────────────
        if self.mode == "RECOVERY":
            elapsed = time.time() - self.recovery_start

            if elapsed < 1.0:
                # Phase 1: reverse briefly to regain manoeuvrability
                cmd.linear.x = -0.1

            elif elapsed < 3.0:
                # Phase 2: rotate toward the direction the target was last seen
                cmd.angular.z = 0.4 if self.last_angle > 0 else -0.4

            else:
                # Phase 3: recovery failed – start a fresh search
                self.search_start = time.time()
                self._set_mode("SEARCHING", "recovery timed out")

            self.cmd_pub.publish(cmd)
            return

        # ── Transition: non-EXPLORE state → SEARCHING on detection timeout ────
        if self.mode not in ("EXPLORE", "DONE", "INIT_SCAN") and time_since_seen > self.target_timeout:
            self.search_start = time.time()
            self._set_mode("SEARCHING", "detection timeout")

        # ── INIT_SCAN ────────────────────────────────────────────────────────
        if self.mode == "INIT_SCAN":
            elapsed = time.time() - self.init_scan_start

            if elapsed >= INIT_SCAN_DURATION:
                # Full revolution complete with no detection → begin spiral search
                self.search_start = time.time()
                self._set_mode("SEARCHING", "360 ° sweep complete, no target found")
            else:
                # Spin in place at a constant angular rate
                remaining = INIT_SCAN_DURATION - elapsed
                self.get_logger().debug(
                    f"INIT_SCAN: {math.degrees(INIT_SCAN_ANGULAR * elapsed):.0f} ° "
                    f"/ 360 °  ({remaining:.1f} s remaining)"
                )
                cmd.angular.z = INIT_SCAN_ANGULAR
                self.cmd_pub.publish(cmd)
            return

        # ── SEARCHING ────────────────────────────────────────────────────────
        if self.mode == "SEARCHING":
            t = time.time() - self.search_start

            # Angular speed decays so the search spiral gradually widens
            angular = self.search_angular_start - self.search_expand_rate * t
            angular = max(angular, self.search_angular_min)

            cmd.linear.x = self.search_linear
            cmd.angular.z = angular
            self.cmd_pub.publish(cmd)
            return

        # ── Guard: no valid detection yet ────────────────────────────────────
        if self.target_angle is None:
            self.cmd_pub.publish(cmd)  # publish zero-velocity (safe default)
            return

        angle_error = self.target_angle

        # ── Transition: TRACKING → EXPLORE when target is close enough ────────
        if self.mode == "TRACKING" and self.front_distance < self.stop_distance:
            self.explore_mode = "LEFT"
            self.explore_start = time.time()
            self.best_size = self.target_size
            self._set_mode(
                "EXPLORE",
                f"distance {self.front_distance:.2f} m < {self.stop_distance} m",
            )

        # ── EXPLORE ──────────────────────────────────────────────────────────
        if self.mode == "EXPLORE":
            self._run_explore(cmd, angle_error)
            return

        # ── TRACKING: steer and drive toward target ───────────────────────────
        if abs(angle_error) > self.angle_threshold:
            # Angle too large to drive safely – rotate in place first
            cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR)
        else:
            # Angle small – drive forward with a gentle steering correction
            cmd.linear.x = TRACK_LINEAR
            cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 2 / 3)

        self.cmd_pub.publish(cmd)

    # -- EXPLORE sub-state machine -------------------------------------------

    def _run_explore(self, cmd: Twist, angle_error: float) -> None:
        """
        Execute one tick of the EXPLORE lateral-scan sequence.

        The full sequence visits three positions – left, centre, right – and
        re-centres the camera on the target at each measurement point before
        recording the bbox area.  After the sweep the robot returns to the
        position that gave the largest bbox area (i.e. best viewing angle)
        and re-centres one final time before transitioning to DONE.

        Position reference (all relative to where EXPLORE was entered):
          centre  0
          left   –1 step   (one CRAB_STEP_DURATION of –y movement)
          right  +1 step   (one CRAB_STEP_DURATION of +y movement)

        Sub-state sequence
        ──────────────────
          LEFT          strafe –y for 1 × CRAB_STEP_DURATION
          LEFT_ALIGN    rotate until |angle_error| < ALIGN_TOLERANCE; record left_size
          RIGHT         strafe +y for 2 × CRAB_STEP_DURATION  (left→centre→right)
          RIGHT_ALIGN   rotate until |angle_error| < ALIGN_TOLERANCE; record right_size
          DECIDE        compare sizes; choose return sub-state
          RETURN_CENTER strafe –y 1 × step  (right→centre); then FINAL_ALIGN
          RETURN_LEFT   strafe –y 2 × steps (right→centre→left); then FINAL_ALIGN
          RETURN_RIGHT  already at right; transition directly to FINAL_ALIGN
          FINAL_ALIGN   rotate until centred; then → DONE
        """
        elapsed = time.time() - self.explore_start

        def _set_explore(sub: str, log: str = "") -> None:
            """Transition the EXPLORE sub-state and log the change."""
            suffix = f"  – {log}" if log else ""
            self.get_logger().info(f"EXPLORE  {self.explore_mode}  →  {sub}{suffix}")
            self.explore_mode = sub
            self.explore_start = time.time()

        # ── LEFT: strafe one step to the left ────────────────────────────────
        if self.explore_mode == "LEFT":
            cmd.linear.y = -CRAB_SPEED

            if elapsed >= CRAB_STEP_DURATION:
                _set_explore("LEFT_ALIGN", "left step done")

        # ── LEFT_ALIGN: hold position; rotate until centred ──────────────────
        elif self.explore_mode == "LEFT_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                self.left_size = self.target_size
                _set_explore("RIGHT", f"left_size = {self.left_size:.3f}")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        # ── RIGHT: strafe two steps to the right (left → centre → right) ─────
        elif self.explore_mode == "RIGHT":
            cmd.linear.y = CRAB_SPEED

            if elapsed >= 2 * CRAB_STEP_DURATION:
                _set_explore("RIGHT_ALIGN", "right step done")

        # ── RIGHT_ALIGN: hold position; rotate until centred ─────────────────
        elif self.explore_mode == "RIGHT_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                self.right_size = self.target_size
                _set_explore("DECIDE", f"right_size = {self.right_size:.3f}")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        # ── DECIDE: compare the three measurements ────────────────────────────
        elif self.explore_mode == "DECIDE":
            self.get_logger().info(
                f"EXPLORE DECIDE:  centre={self.best_size:.3f}  "
                f"left={self.left_size:.3f}  right={self.right_size:.3f}"
            )
            if self.best_size >= self.left_size and self.best_size >= self.right_size:
                _set_explore("RETURN_CENTER", "centre is best")
            elif self.left_size >= self.right_size:
                _set_explore("RETURN_LEFT", "left is best")
            else:
                _set_explore("FINAL_ALIGN", "right is best – already here")

        # ── RETURN_CENTER: strafe left one step (right → centre) ─────────────
        elif self.explore_mode == "RETURN_CENTER":
            cmd.linear.y = -CRAB_SPEED

            if elapsed >= CRAB_STEP_DURATION:
                _set_explore("FINAL_ALIGN", "at centre")

        # ── RETURN_LEFT: strafe left two steps (right → centre → left) ───────
        elif self.explore_mode == "RETURN_LEFT":
            cmd.linear.y = -CRAB_SPEED

            if elapsed >= 2 * CRAB_STEP_DURATION:
                _set_explore("FINAL_ALIGN", "at left")

        # ── FINAL_ALIGN: re-centre camera, then signal task complete ──────────
        elif self.explore_mode == "FINAL_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                self._set_mode("DONE", "EXPLORE complete")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        self.cmd_pub.publish(cmd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, limit: float) -> float:
    """Return *value* clamped to the symmetric range [-limit, +limit]."""
    return max(-limit, min(limit, value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """Initialise ROS 2, spin the node, and clean up on exit."""
    rclpy.init(args=args)
    node = VisionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()