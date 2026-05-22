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
  LEFT           – strafe left for one step
  LEFT_ALIGN     – stop and rotate until camera is centred; record left_size
  RIGHT          – strafe right two steps (left → centre → right)
  RIGHT_ALIGN    – stop and rotate until camera is centred; record right_size
  DECIDE         – compare bbox areas; choose best lateral position:
                     • centre won  → RETURN_CENTER → FINAL_ALIGN → DONE
                     • left won    → RETURN_LEFT   → RECENTER_ALIGN → restart
                     • right won   → RECENTER_ALIGN (already there) → restart
  RETURN_CENTER  – strafe right one step back to centre; then FINAL_ALIGN
  RETURN_LEFT    – strafe left two steps (right → centre → left); then RECENTER_ALIGN
  RECENTER_ALIGN – re-align camera at the new centre position, record fresh
                   best_size, then restart from LEFT (loop continues until
                   centre wins)
  FINAL_ALIGN    – rotate until camera is centred; then → DONE

The EXPLORE loop therefore keeps shifting the "centre" toward the direction
with the largest bounding box on every iteration.  It terminates only when
the centre position scores higher than both neighbours.

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

# Angle tolerance for "camera is centred" checks (rad)
ALIGN_TOLERANCE: float = math.radians(1)

# Angular speed and total duration of the initial 360 ° scan
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
    scan (EXPLORE) that repeats until the centre position yields the largest
    bounding-box area (i.e. the robot has found its optimal viewing angle).
    The camera is re-centred at every measurement point for consistency.
    """

    # -- Lifecycle -----------------------------------------------------------

    def __init__(self) -> None:
        super().__init__("vision_controller")

        # ── Publishers ───────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist,
            "/mirte_base_controller/cmd_vel_unstamped",
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            "/state_change",
            10,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
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

        self.create_subscription(
            String,
            "/robot_state",
            self._state_callback,
            10,
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
        self.mode: str = "INIT_SCAN"
        self.init_scan_start: float = time.time()
        self.search_start: float = 0.0
        self.recovery_start: float | None = None
        self.robot_state: str = "IDLE"          # latest value from /robot_state
        self.get_logger().info("Mode → INIT_SCAN  (performing 360 ° sweep)")

        # ── EXPLORE sub-state ────────────────────────────────────────────────
        self.explore_mode: str | None = None  # current EXPLORE sub-state
        self.explore_start: float = 0.0       # epoch of current sub-state start
        self.best_size: float = 0.0           # bbox area at current centre
        self.left_size: float = 0.0           # bbox area measured at left step
        self.right_size: float = 0.0          # bbox area measured at right step
        self._explore_iteration: int = 0      # counts EXPLORE restarts (for logging)

        # ── Main control loop (10 Hz) ────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)

    # -- Mode helpers --------------------------------------------------------

    def _set_mode(self, new_mode: str, extra: str = "") -> None:
        """
        Transition to *new_mode*, logging the change.

        Parameters
        ----------
        new_mode:
            Target state name (e.g. ``"TRACKING"``).
        extra:
            Optional context appended to the log message.
        """
        if new_mode == self.mode:
            return
        suffix = f"  ({extra})" if extra else ""
        self.get_logger().info(f"Mode  {self.mode}  →  {new_mode}{suffix}")
        self.mode = new_mode

    def _set_explore(self, sub: str, log: str = "") -> None:
        """
        Transition the EXPLORE sub-state, reset the sub-state timer, and log.

        Parameters
        ----------
        sub:
            Target EXPLORE sub-state name.
        log:
            Optional context string appended to the log message.
        """
        suffix = f"  – {log}" if log else ""
        self.get_logger().info(
            f"EXPLORE [{self._explore_iteration}]  "
            f"{self.explore_mode}  →  {sub}{suffix}"
        )
        self.explore_mode = sub
        self.explore_start = time.time()

    # -- Callbacks -----------------------------------------------------------

    def _angle_callback(self, msg: String) -> None:
        """
        Receive the latest detection from the vision pipeline.

        JSON fields
        ───────────
          angle     – horizontal angle to target centre (rad);
                      positive = target is LEFT of image centre
          bbox_area – bounding-box pixel area (grows as robot gets closer)
        """
        data = json.loads(msg.data)
        angle: float = data["angle"]

        self.target_angle = angle
        self.last_angle = angle
        self.last_detection_time = time.time()
        self.target_size = data["bbox_area"]

        # Never interrupt an ongoing EXPLORE or a completed task
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

    def _state_callback(self, msg: String) -> None:
        """Track the current robot state published by StateManager."""
        self.robot_state = msg.data

    # -- Control loop --------------------------------------------------------

    def _control_loop(self) -> None:
        """
        10 Hz state-machine tick.

        State priority (highest first):
          DONE      – task finished; robot stopped; DONE published upstream
          RECOVERY  – brief reverse then spin toward last known angle
          INIT_SCAN – one full 360 ° rotation before the outward spiral
          SEARCHING – forward spiral until a target appears
          EXPLORE   – iterative lateral scan with realignment (repeats until
                      centre position wins)
          TRACKING  – proportional angular + forward drive toward target
        """
        cmd = Twist()
        time_since_seen = time.time() - self.last_detection_time

        # Gate: only act when the state manager says it's our turn
        if self.robot_state != "TRACK_WHITEBOARD":
            return

        # ── DONE ─────────────────────────────────────────────────────────────
        if self.mode == "DONE":
            self.cmd_pub.publish(Twist())
            done_msg = String()
            done_msg.data = "DONE"
            self.state_pub.publish(done_msg)
            return

        # ── Transition: TRACKING → RECOVERY on detection timeout ─────────────
        if self.mode == "TRACKING" and time_since_seen > self.target_timeout:
            self._set_mode("RECOVERY", "target lost")
            self.recovery_start = time.time()

        # ── RECOVERY ─────────────────────────────────────────────────────────
        if self.mode == "RECOVERY":
            elapsed = time.time() - self.recovery_start

            if elapsed < 1.0:
                cmd.linear.x = -0.1                          # phase 1: reverse
            elif elapsed < 3.0:
                cmd.angular.z = 0.4 if self.last_angle > 0 else -0.4  # phase 2: spin
            else:
                self.search_start = time.time()
                self._set_mode("SEARCHING", "recovery timed out")

            self.cmd_pub.publish(cmd)
            return

        # ── Transition: non-EXPLORE state → SEARCHING on detection timeout ────
        if (
            self.mode not in ("EXPLORE", "DONE", "INIT_SCAN")
            and time_since_seen > self.target_timeout
        ):
            self.search_start = time.time()
            self._set_mode("SEARCHING", "detection timeout")

        # ── INIT_SCAN ────────────────────────────────────────────────────────
        if self.mode == "INIT_SCAN":
            elapsed = time.time() - self.init_scan_start

            if elapsed >= INIT_SCAN_DURATION:
                self.search_start = time.time()
                self._set_mode("SEARCHING", "360 ° sweep complete, no target found")
            else:
                self.get_logger().debug(
                    f"INIT_SCAN: {math.degrees(INIT_SCAN_ANGULAR * elapsed):.0f} ° / 360 °"
                )
                cmd.angular.z = INIT_SCAN_ANGULAR
                self.cmd_pub.publish(cmd)
            return

        # ── SEARCHING ────────────────────────────────────────────────────────
        if self.mode == "SEARCHING":
            t = time.time() - self.search_start
            angular = self.search_angular_start - self.search_expand_rate * t
            angular = max(angular, self.search_angular_min)
            cmd.linear.x = self.search_linear
            cmd.angular.z = angular
            self.cmd_pub.publish(cmd)
            return

        # ── Guard: no valid detection yet ────────────────────────────────────
        if self.target_angle is None:
            self.cmd_pub.publish(cmd)
            return

        angle_error = self.target_angle

        # ── Transition: TRACKING → EXPLORE when target is close enough ────────
        if self.mode == "TRACKING" and self.front_distance < self.stop_distance:
            self._explore_iteration = 0
            self.explore_mode = "LEFT"
            self.explore_start = time.time()
            self.best_size = self.target_size
            self.left_size = 0.0
            self.right_size = 0.0
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
            cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR)
        else:
            cmd.linear.x = TRACK_LINEAR
            cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 2 / 3)

        self.cmd_pub.publish(cmd)

    # -- EXPLORE sub-state machine -------------------------------------------

    def _run_explore(self, cmd: Twist, angle_error: float) -> None:
        """
        Execute one tick of the iterative EXPLORE lateral-scan sequence.

        The sequence performs a left / right sweep from the current centre
        position, re-centring the camera at each measurement point for a
        consistent comparison.  If the centre does not win, the robot moves
        to the winning position, re-aligns (RECENTER_ALIGN), records a fresh
        best_size for that new centre, and restarts the sweep.  This repeats
        until the centre position scores higher than both neighbours.

        Position reference (relative to current "centre"):
          centre  0
          left   -1 step  (CRAB_STEP_DURATION at -y)
          right  +1 step  (CRAB_STEP_DURATION at +y)

        Sub-state sequence
        ──────────────────
          LEFT           strafe -y for 1 × CRAB_STEP_DURATION
          LEFT_ALIGN     rotate until aligned; record left_size
          RIGHT          strafe +y for 2 × CRAB_STEP_DURATION
          RIGHT_ALIGN    rotate until aligned; record right_size
          DECIDE         compare; pick winner:
                           centre → RETURN_CENTER → FINAL_ALIGN → DONE
                           left   → RETURN_LEFT   → RECENTER_ALIGN → restart
                           right  → RECENTER_ALIGN (already here) → restart
          RETURN_CENTER  strafe -y 1 × step (right → centre)
          RETURN_LEFT    strafe -y 2 × steps (right → centre → left)
          RECENTER_ALIGN rotate until aligned; record fresh best_size; go to LEFT
          FINAL_ALIGN    rotate until aligned; → DONE
        """
        elapsed = time.time() - self.explore_start

        # ── LEFT ─────────────────────────────────────────────────────────────
        if self.explore_mode == "LEFT":
            cmd.linear.y = -CRAB_SPEED
            if elapsed >= CRAB_STEP_DURATION:
                self._set_explore("LEFT_ALIGN", "left step done")

        # ── LEFT_ALIGN ───────────────────────────────────────────────────────
        elif self.explore_mode == "LEFT_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                self.left_size = self.target_size
                self._set_explore("RIGHT", f"left_size = {self.left_size:.3f}")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        # ── RIGHT ────────────────────────────────────────────────────────────
        elif self.explore_mode == "RIGHT":
            cmd.linear.y = CRAB_SPEED
            if elapsed >= 2 * CRAB_STEP_DURATION:
                self._set_explore("RIGHT_ALIGN", "right step done")

        # ── RIGHT_ALIGN ──────────────────────────────────────────────────────
        elif self.explore_mode == "RIGHT_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                self.right_size = self.target_size
                self._set_explore("DECIDE", f"right_size = {self.right_size:.3f}")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        # ── DECIDE ───────────────────────────────────────────────────────────
        elif self.explore_mode == "DECIDE":
            self.get_logger().info(
                f"EXPLORE [{self._explore_iteration}] DECIDE:  "
                f"centre={self.best_size:.3f}  "
                f"left={self.left_size:.3f}  "
                f"right={self.right_size:.3f}"
            )

            # Centre is the best viewpoint – head back and finish
            if self.best_size >= self.left_size and self.best_size >= self.right_size:
                self._set_explore("RETURN_CENTER", "centre is best → finishing")

            # Left is better – move there and restart the sweep from that new centre
            elif self.left_size >= self.right_size:
                self._set_explore("RETURN_LEFT", "left is best → recentring there")

            # Right is better – already here; realign and restart from this new centre
            else:
                self._set_explore("RECENTER_ALIGN", "right is best → recentring here")

        # ── RETURN_CENTER ────────────────────────────────────────────────────
        # Currently at the right position; strafe one step left to reach centre.
        elif self.explore_mode == "RETURN_CENTER":
            cmd.linear.y = -CRAB_SPEED
            if elapsed >= CRAB_STEP_DURATION:
                self._set_explore("FINAL_ALIGN", "at centre")

        # ── RETURN_LEFT ──────────────────────────────────────────────────────
        # Currently at the right position; strafe two steps left to reach left.
        elif self.explore_mode == "RETURN_LEFT":
            cmd.linear.y = -CRAB_SPEED
            if elapsed >= 2 * CRAB_STEP_DURATION:
                self._set_explore("RECENTER_ALIGN", "at left position")

        # ── RECENTER_ALIGN ───────────────────────────────────────────────────
        # Arrived at the new centre position (either left or right won).
        # Re-align the camera, record a fresh best_size, then restart the sweep.
        elif self.explore_mode == "RECENTER_ALIGN":
            if abs(angle_error) < ALIGN_TOLERANCE:
                # Record current position as the new centre baseline
                self.best_size = self.target_size
                self.left_size = 0.0
                self.right_size = 0.0
                self._explore_iteration += 1
                self.get_logger().info(
                    f"EXPLORE: new centre baseline = {self.best_size:.3f}  "
                    f"(iteration {self._explore_iteration})"
                )
                # Restart the sweep from this new centre
                self._set_explore("LEFT", "restarting sweep")
            else:
                cmd.angular.z = _clamp(ANGLE_KP * angle_error, MAX_ANGULAR * 0.5)

        # ── FINAL_ALIGN ──────────────────────────────────────────────────────
        # Centre won; re-align one last time then declare the task complete.
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