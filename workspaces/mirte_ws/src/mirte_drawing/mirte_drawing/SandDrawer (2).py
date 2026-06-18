"""
sand_drawer.py
--------------
ROS 2 node that draws text in sand by controlling the Mirte Master arm.

Flow
────
  IDLE
    │  DRAW_PATTERN received on /robot_state
    ▼
  PROBING  – arm steps down until joint effort exceeds EFFORT_THRESHOLD;
             records wrist height from current joint angles
    │  contact detected
    ▼
  DRAWING  – converts text from /whiteboard_text to arm waypoints using a
             built-in single-stroke font; sends full trajectory at once;
             pen lifts between letter strokes
    │  trajectory timer fires
    ▼
  IDLE  (publishes DONE on /state_change)

Subscriptions
  /robot_state     (std_msgs/String)        – pipeline state gate
  /whiteboard_text (std_msgs/String)        – JSON: {"text": "HELLO", ...}
  /target_angle    (std_msgs/String)        – JSON: {"angle_x": …, "angle_y": …,
                                              "bbox_area": <px²>}; bbox_area is
                                              used to scale letters to the sandbox
  /joint_states    (sensor_msgs/JointState) – effort feedback for probing
  /odom            (nav_msgs/Odometry)      – closed-loop strafe control

Publications
  /state_change    (std_msgs/String)       – DONE when drawing complete
  /mirte_master_arm_controller/joint_trajectory
                   (trajectory_msgs/JointTrajectory)
  /cmd_vel         (geometry_msgs/Twist)   – base strafe commands

Letter sizing
─────────────
  At the start of each draw the node calls _compute_letter_sizes(), which
  converts the last received bbox_area (pixels²) to a sandbox dimension in
  metres using the camera FOV (60°) and the robot's stopping distance
  (SANDBOX_STOP_DIST = 0.5 m).  Letter width is chosen so that the entire
  word fits across the sandbox; letter height is chosen so that one letter
  fits in the sandbox depth.  Both dimensions are clamped to
  [MIN_LETTER_SIZE, MAX_LETTER_HEIGHT / MAX_LETTER_WIDTH] to keep every
  waypoint within arm reach.  If no bbox_area has been received the
  hardcoded defaults (LETTER_HEIGHT / LETTER_WIDTH) are used instead.
  Tune SANDBOX_SCALE_FACTOR if bbox_area is the ArUco marker area rather
  than the full sandbox area.

Kinematics (verified against the real Mirte Master URDF)
────────────────────────────────────────────────────────
  Chain: frame_link → shoulder_pan → shoulder_lift → elbow → wrist
  All arm joints are limited to ±90° (pi/2). This is the single most
  important constraint: any IK solution outside that range is rejected by
  the controller and the arm freezes in a wrong pose (e.g. up in the air).

  shoulder_lift = 0  → first link points straight UP (+z).
  Angles are measured FROM VERTICAL, so:
      horizontal reach = L*sin(angle)   vertical = L*cos(angle)

  The shoulder pivot sits SHOULDER_Z (=0.0881 m) above frame_link, while
  the ground is at GROUND_Z (=-0.0955 m). Because of the ±90° limits the
  WRIST cannot descend below z≈-0.054 m, i.e. it stays ~4 cm above the
  ground. The pen/gripper mounted below the wrist bridges that last gap,
  so we keep the wrist at WRIST_DRAW_HEIGHT and let the pen touch the sand.

  Direction: shoulder_pan's zero points to the BACK and it only swings ±90°,
  so reaching the robot FRONT uses the mirrored fold (negative reach, with
  negative lift/elbow). Letters are traced in a band roughly 0.27–0.33 m in
  FRONT of the robot. For longer words the base strafes between letters
  instead of widening the reach.

Calibration – adjust for your robot/sim
  WRIST_DRAW_HEIGHT   – wrist z (frame_link) while drawing; lower it if the
                        pen does not reach the sand, raise it if it digs in
  WRIST_DRAW_ANGLE    – wrist_joint angle that points the pen down
  PEN_LIFT            – how much to raise the wrist between strokes
  L1, L2              – link lengths (match the URDF: 0.1378 / 0.14265)
  EFFORT_THRESHOLD    – press arm gently on a surface and read /joint_states
  SANDBOX_STOP_DIST   – LiDAR distance at which the robot stops before the sandbox
  SANDBOX_SCALE_FACTOR – increase if bbox_area is the marker, not the full sandbox
"""

import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ---------------------------------------------------------------------------
# Arm geometry – taken directly from the Mirte Master URDF (arm.xacro)
# ---------------------------------------------------------------------------

L1: float = 0.1378            # shoulder_lift → elbow link length (m)
L2: float = 0.14265           # elbow → wrist link length (m)

SHOULDER_Z: float = 0.0881    # shoulder_lift pivot height above frame_link (m)
SHOULDER_Y: float = 0.079274  # forward offset from frame_link origin to pivot (m)
REACH_OFFSET: float = 0.00625 # small horizontal offset baked into the chain (m)

JOINT_LIMIT: float = math.pi / 2   # ±90° hard limit on every arm joint (rad)
GROUND_Z: float = -0.0955     # ground plane in frame_link coordinates (m)

EFFORT_THRESHOLD: float = 0.5 # joint effort (Nm) that signals sand contact

# ---------------------------------------------------------------------------
# Drawing height / pen
# ---------------------------------------------------------------------------
# Set to True to skip probing and draw at a fixed wrist height (for testing).
SKIP_PROBING: bool = True

# Wrist z (frame_link) held while drawing. There is no sand, so the letters
# are traced a few cm ABOVE the floor. A pen mounted where the gripper fingers
# were (≈2.5 cm long, pointing the finger direction) sits ~6.5 cm below the
# wrist, so at WRIST_DRAW_HEIGHT=0 the pen tip clears the ground by ~3 cm.
# Pen-tip height tracks this value ≈1:1:
#   draw nearer the floor → lower WRIST_DRAW_HEIGHT (≈-0.03 would touch)
#   draw higher up        → raise it
PEN_LENGTH:        float = 0.025  # pen length below the gripper mount (m), informational
WRIST_DRAW_HEIGHT: float = 0.0
WRIST_DRAW_ANGLE:  float = 0.0    # wrist_joint angle to aim the pen downward (rad)

PEN_LIFT: float = 0.015       # how much to raise the wrist between strokes (m)

# Probing lowers the wrist height at the draw centre until effort spikes.
PROBE_Z_START: float = 0.05   # wrist height to start probing from (m)
PROBE_Z_STEP:  float = 0.01   # how much to lower the wrist each probe step (m)
PROBE_Z_MIN:   float = -0.05  # abort if probe reaches this height without contact
PROBE_INTERVAL: float = 0.5   # seconds between probe steps

# ---------------------------------------------------------------------------
# Drawing layout  (sized to the reachable band – keep letters small!)
# ---------------------------------------------------------------------------

DRAW_X_CENTER:  float = 0.300 # center forward distance of drawing area (m)
LETTER_HEIGHT:  float = 0.055 # letter extent in the forward direction (m)
LETTER_WIDTH:   float = 0.050 # letter extent in the lateral direction (m)
LETTER_GAP:     float = 0.015 # gap between letters (m)
DRAW_STEP_SEC:  int   = 2     # seconds per waypoint

# Center-to-center spacing between consecutive letters (m). Each letter is
# drawn centred in front of the arm; the base strafes this exact distance
# between letters, so a word of any length stays within the arm's reach.
LETTER_PITCH:   float = LETTER_WIDTH + LETTER_GAP

# ---------------------------------------------------------------------------
# Sandbox-aware letter sizing
# ---------------------------------------------------------------------------
# bbox_area (pixels²) from /target_angle is converted to metres using the
# camera FOV and the robot's stopping distance during TRACK_SANDPIT.
# SANDBOX_SCALE_FACTOR: set to 1.0 if bbox_area is the sandbox area in pixels;
# increase it (e.g. 5.0) if bbox_area is only the ArUco marker area and the
# sandbox is larger than the marker.
SANDBOX_STOP_DIST:    float = 0.5          # LiDAR stop distance for TRACK_SANDPIT (m)
SANDBOX_FOV_RAD:      float = math.pi / 3  # camera horizontal FOV – matches ArucoDetectionSand
SANDBOX_IMG_WIDTH:    int   = 640          # camera image width – matches ArucoDetectionSand
SANDBOX_SCALE_FACTOR: float = 1.0          # scale bbox_area→sandbox; tune if needed
SANDBOX_MARGIN:       float = 0.85         # fraction of sandbox dimension used for drawing
# Hard limits to keep letters within arm reach
MAX_LETTER_HEIGHT:    float = 0.060        # max letter height – forward direction (m)
MAX_LETTER_WIDTH:     float = 0.090        # max letter width  – lateral direction (m)
MIN_LETTER_SIZE:      float = 0.015        # minimum letter dimension (m)

# ---------------------------------------------------------------------------
# Base motion – per-letter sideways shift (mecanum strafe)
# ---------------------------------------------------------------------------

# Base velocity topic. This Gazebo sim exposes /cmd_vel (geometry_msgs/Twist);
# the real robot / other setups may use /mirte_base_controller/cmd_vel_unstamped.
BASE_CMD_TOPIC: str   = "/cmd_vel"
ODOM_TOPIC:     str   = "/odom"
STRAFE_SPEED:   float = 0.05  # base strafe speed (m/s)
STRAFE_SIGN:    float = -1.0  # +1 = base moves left (+y); -1 = right. Flip if word is mirrored
STRAFE_TIMEOUT: float = 15.0  # safety: max seconds for one strafe before giving up
TRAVEL_LIFT:    float = 0.015 # extra wrist lift while the base is moving (m)
# Closed-loop uses /odom to strafe exactly LETTER_PITCH. Set False if the base
# strafes but odom does not report lateral motion (then it drives open-loop for
# distance / STRAFE_SPEED seconds instead).
STRAFE_CLOSED_LOOP: bool = True

# ---------------------------------------------------------------------------
# Single-stroke font
# ---------------------------------------------------------------------------
# Each letter is a list of strokes.
# Each stroke is a list of (x, y) points, normalised to [0-1] × [0-1].
#   x = 0 → left edge,   x = 1 → right edge  (maps to robot lateral axis)
#   y = 0 → bottom,      y = 1 → top          (maps to robot forward axis)

STROKES: dict[str, list[list[tuple[float, float]]]] = {
    'A': [[(0.0,0.0),(0.5,1.0),(1.0,0.0)],
          [(0.2,0.4),(0.8,0.4)]],
    'B': [[(0.0,0.0),(0.0,1.0),(0.7,1.0),(0.9,0.85),(0.7,0.5),(0.0,0.5)],
          [(0.0,0.5),(0.7,0.5),(0.9,0.35),(0.7,0.0),(0.0,0.0)]],
    'C': [[(0.9,0.85),(0.7,1.0),(0.3,1.0),(0.0,0.75),(0.0,0.25),(0.3,0.0),(0.7,0.0),(0.9,0.15)]],
    'D': [[(0.0,0.0),(0.0,1.0),(0.6,1.0),(0.9,0.75),(0.9,0.25),(0.6,0.0),(0.0,0.0)]],
    'E': [[(1.0,1.0),(0.0,1.0),(0.0,0.0),(1.0,0.0)],
          [(0.0,0.5),(0.7,0.5)]],
    'F': [[(0.0,0.0),(0.0,1.0),(1.0,1.0)],
          [(0.0,0.5),(0.7,0.5)]],
    'G': [[(0.9,0.85),(0.7,1.0),(0.3,1.0),(0.0,0.75),(0.0,0.25),(0.3,0.0),(0.7,0.0),(0.9,0.25),(0.9,0.5),(0.5,0.5)]],
    'H': [[(0.0,0.0),(0.0,1.0)],
          [(0.0,0.5),(1.0,0.5)],
          [(1.0,1.0),(1.0,0.0)]],
    'I': [[(0.2,1.0),(0.8,1.0)],
          [(0.5,1.0),(0.5,0.0)],
          [(0.2,0.0),(0.8,0.0)]],
    'J': [[(0.2,0.25),(0.2,0.0),(0.8,0.0),(0.8,1.0)]],
    'K': [[(0.0,0.0),(0.0,1.0)],
          [(1.0,1.0),(0.0,0.5),(1.0,0.0)]],
    'L': [[(0.0,1.0),(0.0,0.0),(1.0,0.0)]],
    'M': [[(0.0,0.0),(0.0,1.0),(0.5,0.5),(1.0,1.0),(1.0,0.0)]],
    'N': [[(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]],
    'O': [[(0.3,0.0),(0.7,0.0),(0.9,0.25),(0.9,0.75),(0.7,1.0),(0.3,1.0),(0.1,0.75),(0.1,0.25),(0.3,0.0)]],
    'P': [[(0.0,0.0),(0.0,1.0),(0.7,1.0),(0.9,0.85),(0.9,0.65),(0.7,0.5),(0.0,0.5)]],
    'Q': [[(0.3,0.0),(0.7,0.0),(0.9,0.25),(0.9,0.75),(0.7,1.0),(0.3,1.0),(0.1,0.75),(0.1,0.25),(0.3,0.0)],
          [(0.6,0.2),(1.0,0.0)]],
    'R': [[(0.0,0.0),(0.0,1.0),(0.7,1.0),(0.9,0.85),(0.9,0.65),(0.7,0.5),(0.0,0.5)],
          [(0.5,0.5),(1.0,0.0)]],
    'S': [[(0.9,0.85),(0.7,1.0),(0.3,1.0),(0.1,0.85),(0.1,0.65),(0.5,0.5),(0.9,0.35),(0.9,0.15),(0.7,0.0),(0.3,0.0),(0.1,0.15)]],
    'T': [[(0.0,1.0),(1.0,1.0)],
          [(0.5,1.0),(0.5,0.0)]],
    'U': [[(0.0,1.0),(0.0,0.25),(0.3,0.0),(0.7,0.0),(1.0,0.25),(1.0,1.0)]],
    'V': [[(0.0,1.0),(0.5,0.0),(1.0,1.0)]],
    'W': [[(0.0,1.0),(0.25,0.0),(0.5,0.5),(0.75,0.0),(1.0,1.0)]],
    'X': [[(0.0,1.0),(1.0,0.0)],
          [(1.0,1.0),(0.0,0.0)]],
    'Y': [[(0.0,1.0),(0.5,0.5),(1.0,1.0)],
          [(0.5,0.5),(0.5,0.0)]],
    'Z': [[(0.0,1.0),(1.0,1.0),(0.0,0.0),(1.0,0.0)]],
    ' ': [],
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SandDrawer(Node):

    def __init__(self) -> None:
        super().__init__("sand_drawer")

        # ── Publishers ───────────────────────────────────────────────────────
        self.state_pub = self.create_publisher(String, "/state_change", 10)
        self.arm_pub   = self.create_publisher(
            JointTrajectory,
            "/mirte_master_arm_controller/joint_trajectory",
            10,
        )
        self.base_pub  = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(String,     "/robot_state",     self._state_callback,      10)
        self.create_subscription(String,     "/whiteboard_text", self._text_callback,        10)
        self.create_subscription(String,     "/target_angle",    self._target_callback,      10)
        self.create_subscription(JointState, "/joint_states",    self._joint_state_callback, 10)
        self.create_subscription(Odometry,   ODOM_TOPIC,         self._odom_callback,        10)

        # ── Internal state ───────────────────────────────────────────────────
        self._mode: str             = "IDLE"
        self._text: str             = ""
        self._draw_z: float | None  = None   # wrist height held while drawing

        self._joint_positions: dict[str, float] = {}
        self._joint_efforts:   dict[str, float] = {}
        self._odom_xy: tuple[float, float] | None = None

        self._probe_z: float = PROBE_Z_START
        self._probe_timer = None
        self._done_timer  = None

        # Per-letter drawing / strafing state
        self._letters: list[str] = []
        self._letter_idx: int    = 0
        self._strafe_timer       = None
        self._strafe_start: tuple[float, float] | None = None
        self._strafe_t0: float   = 0.0

        # Sandbox size (pixels²) from /target_angle; used to scale letters
        self._bbox_area: float | None = None
        # Active letter dimensions, recomputed each draw call
        self._cur_letter_height: float = LETTER_HEIGHT
        self._cur_letter_width:  float = LETTER_WIDTH
        self._cur_letter_pitch:  float = LETTER_PITCH

        self._selftest()

    # -- Subscriptions -------------------------------------------------------

    def _state_callback(self, msg: String) -> None:
        if msg.data == "DRAW_PATTERN" and self._mode == "IDLE":
            if not self._text:
                self.get_logger().warn("No text received yet – waiting for /whiteboard_text")
                return
            if SKIP_PROBING:
                self._draw_z = WRIST_DRAW_HEIGHT
                self.get_logger().info(
                    f"DRAW_PATTERN – skipping probe, wrist height={self._draw_z} m, "
                    f"will draw: {self._text!r}"
                )
                self._mode = "DRAWING"
                self._draw()
            else:
                self.get_logger().info(f"DRAW_PATTERN – probing sand, will draw: {self._text!r}")
                self._mode = "PROBING"
                self._start_probe()
        elif msg.data != "DRAW_PATTERN" and self._mode != "IDLE":
            self._cancel_timers()
            self._mode = "IDLE"

    def _text_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._text = data.get("text", "").upper().strip()
            self.get_logger().info(f"Text to draw: {self._text!r}")
        except Exception as exc:
            self.get_logger().warn(f"Text parse error: {exc}")

    def _target_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._bbox_area = float(data["bbox_area"])
        except Exception as exc:
            self.get_logger().warn(f"Target parse error: {exc}")

    def _joint_state_callback(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self._joint_positions[name] = msg.position[i]
            if i < len(msg.effort):
                self._joint_efforts[name] = msg.effort[i]

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    # -- Probing -------------------------------------------------------------
    # Probing lowers the wrist height at the draw centre (via IK) until the
    # joint effort spikes, then records that height as the drawing height.

    def _start_probe(self) -> None:
        self._probe_z = PROBE_Z_START
        self._probe_to(self._probe_z, duration=1)
        self._probe_timer = self.create_timer(PROBE_INTERVAL, self._probe_step)

    def _probe_to(self, z: float, duration: float) -> None:
        sol = self._ik(DRAW_X_CENTER, 0.0, z)
        if sol is None:
            self.get_logger().warn(f"Probe height z={z:.3f} unreachable – aborting")
            self._probe_timer.cancel()
            self._mode = "IDLE"
            return
        pan, lift, elbow, wrist = sol
        self._send_arm(pan=pan, lift=lift, elbow=elbow, wrist=wrist, duration=duration)

    def _probe_step(self) -> None:
        if self._mode != "PROBING":
            self._probe_timer.cancel()
            return

        effort = max(
            abs(self._joint_efforts.get("shoulder_lift_joint", 0.0)),
            abs(self._joint_efforts.get("elbow_joint", 0.0)),
        )

        if effort >= EFFORT_THRESHOLD:
            self._draw_z = self._probe_z
            self.get_logger().info(
                f"Sand contact: effort={effort:.2f} Nm  wrist_z={self._draw_z:.3f} m"
            )
            self._probe_timer.cancel()
            self._mode = "DRAWING"
            self._draw()
            return

        self._probe_z -= PROBE_Z_STEP
        if self._probe_z < PROBE_Z_MIN:
            self.get_logger().warn("Probe reached minimum height without contact")
            self._probe_timer.cancel()
            self._mode = "IDLE"
            return

        self.get_logger().info(
            f"Probing: wrist_z={self._probe_z:.3f} m  effort={effort:.2f}"
        )
        self._probe_to(self._probe_z, duration=PROBE_INTERVAL)

    # -- Drawing -------------------------------------------------------------

    def _draw(self) -> None:
        """Draw the word letter by letter, strafing the base between letters."""
        self._letters = list(self._text)
        self._letter_idx = 0
        self._compute_letter_sizes(len(self._letters))
        self.get_logger().info(
            f"Drawing {len(self._letters)} letters, strafing "
            f"{self._cur_letter_pitch*100:.1f} cm between each: {self._text!r}"
        )
        self._draw_next_letter()

    def _compute_letter_sizes(self, n_letters: int) -> None:
        """Compute letter dimensions that fit the detected sandbox and arm reach.

        Uses bbox_area (px²) from /target_angle, converted to metres via the
        camera FOV and the robot's stopping distance during TRACK_SANDPIT.
        Falls back to the hardcoded defaults when no sandbox data is available.
        """
        if self._bbox_area is None or self._bbox_area <= 0 or n_letters == 0:
            self._cur_letter_height = LETTER_HEIGHT
            self._cur_letter_width  = LETTER_WIDTH
            self._cur_letter_pitch  = LETTER_PITCH
            self.get_logger().info(
                "No sandbox size available – using default letter dimensions"
            )
            return

        # Convert bbox_area (px²) → sandbox dimension (m).
        # sqrt(bbox_area) ≈ sandbox edge in pixels (assumes roughly square sandbox).
        m_per_px = (2.0 * SANDBOX_STOP_DIST * math.tan(SANDBOX_FOV_RAD / 2.0)) / SANDBOX_IMG_WIDTH
        sandbox_dim_m = math.sqrt(self._bbox_area) * m_per_px * SANDBOX_SCALE_FACTOR

        avail = sandbox_dim_m * SANDBOX_MARGIN  # usable sandbox dimension in both axes

        # Height: one letter must fit in the sandbox depth (forward direction).
        h = max(MIN_LETTER_SIZE, min(MAX_LETTER_HEIGHT, avail))

        # Width: the whole word must fit in the sandbox width (lateral direction).
        # word_width = n*w + (n-1)*gap  where gap = (LETTER_GAP/LETTER_WIDTH) * w
        k = LETTER_GAP / LETTER_WIDTH  # keep gap-to-width ratio constant
        denom = n_letters + (n_letters - 1) * k
        w = max(MIN_LETTER_SIZE, min(MAX_LETTER_WIDTH, avail / denom))

        self._cur_letter_height = h
        self._cur_letter_width  = w
        self._cur_letter_pitch  = w + w * k

        self.get_logger().info(
            f"Sandbox: bbox_area={self._bbox_area:.0f} px²  "
            f"→ dim≈{sandbox_dim_m*100:.1f} cm | "
            f"letter: w={w*100:.1f} cm  h={h*100:.1f} cm  "
            f"pitch={self._cur_letter_pitch*100:.1f} cm  n={n_letters}"
        )

    def _draw_next_letter(self) -> None:
        if self._letter_idx >= len(self._letters):
            self._stop_base()
            self._signal_done()
            return

        ch = self._letters[self._letter_idx]
        traj, duration, skipped = self._build_letter_traj(ch)

        if skipped:
            self.get_logger().warn(
                f"Letter {ch!r}: {skipped} waypoints out of reach – skipped"
            )

        if not traj.points:
            # Nothing to draw (space or fully unreachable) – just advance.
            self.get_logger().info(f"Letter {ch!r}: nothing to draw, advancing")
            self._after_letter()
            return

        self.get_logger().info(
            f"Letter {self._letter_idx + 1}/{len(self._letters)} {ch!r}: "
            f"{len(traj.points)} waypoints (~{duration} s)"
        )
        self._stop_base()              # ensure the base holds still while drawing
        self.arm_pub.publish(traj)
        self._done_timer = self.create_timer(float(duration), self._on_letter_complete)

    def _on_letter_complete(self) -> None:
        if self._done_timer:
            self._done_timer.cancel()
            self._done_timer = None
        self._after_letter()

    def _after_letter(self) -> None:
        """Advance to the next letter, strafing the base by one letter pitch."""
        self._letter_idx += 1
        if self._letter_idx >= len(self._letters):
            self._stop_base()
            self._signal_done()
            return
        # Pen is already raised at the end of the letter trajectory; strafe now.
        self._start_strafe(self._cur_letter_pitch)

    def _build_letter_traj(self, ch: str):
        """Build a JointTrajectory for one letter, centred in front of the arm.

        Returns (trajectory, total_seconds, skipped_count). The trajectory
        ends with the pen raised so the base can strafe safely afterwards.
        """
        traj = JointTrajectory()
        traj.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint",        "wrist_joint",
        ]

        strokes = STROKES.get(ch, [])
        waypoints: list[tuple[float, float, float]] = []
        for stroke in strokes:
            # Pen up: move to the first point of the stroke at lift height
            lx, ly = stroke[0]
            fwd, lat = self._letter_to_robot(lx, ly)
            waypoints.append((fwd, lat, self._draw_z + PEN_LIFT))
            # Pen down: draw each point in the stroke
            for lx, ly in stroke:
                fwd, lat = self._letter_to_robot(lx, ly)
                waypoints.append((fwd, lat, self._draw_z))

        if waypoints:
            # End raised and centred, ready for the base to strafe.
            waypoints.append((DRAW_X_CENTER, 0.0, self._draw_z + PEN_LIFT + TRAVEL_LIFT))

        t = DRAW_STEP_SEC
        skipped = 0
        for fwd, lat, z in waypoints:
            result = self._ik(fwd, lat, z)
            if result is None:
                skipped += 1
                continue
            pan, lift, elbow, wrist = result
            wp = JointTrajectoryPoint()
            wp.positions = [pan, lift, elbow, wrist]
            wp.time_from_start.sec = t
            traj.points.append(wp)
            t += DRAW_STEP_SEC

        return traj, t, skipped

    def _letter_to_robot(self, lx: float, ly: float) -> tuple[float, float]:
        """
        Map normalised letter coordinates to robot (forward, lateral), with the
        letter centred laterally in front of the arm.
          lx [0-1]: horizontal in letter  → robot lateral axis (centred)
          ly [0-1]: vertical in letter    → robot forward axis
        The lateral axis is flipped (0.5 - lx) so letters read un-mirrored from
        the front; if the word order comes out reversed, flip STRAFE_SIGN.
        """
        fwd = DRAW_X_CENTER + (ly - 0.5) * self._cur_letter_height
        lat = (0.5 - lx) * self._cur_letter_width
        return fwd, lat

    # -- Base strafing -------------------------------------------------------

    def _start_strafe(self, distance: float) -> None:
        """Strafe the base sideways by `distance` metres (closed-loop on odom)."""
        if self._odom_xy is None:
            self.get_logger().warn(
                f"No odom on {ODOM_TOPIC}; strafing open-loop for "
                f"{distance/STRAFE_SPEED:.1f} s instead"
            )
        self._strafe_start = self._odom_xy
        self._strafe_t0 = self._now()
        self._strafe_timer = self.create_timer(0.05, lambda: self._strafe_step(distance))

    def _strafe_step(self, distance: float) -> None:
        if self._mode != "DRAWING":
            self._stop_base()
            if self._strafe_timer:
                self._strafe_timer.cancel()
                self._strafe_timer = None
            return

        elapsed = self._now() - self._strafe_t0

        use_odom = (STRAFE_CLOSED_LOOP and self._strafe_start is not None
                    and self._odom_xy is not None)
        if use_odom:
            moved = math.dist(self._odom_xy, self._strafe_start)
            done = moved >= distance
        else:
            # open-loop: drive for distance / speed seconds
            done = elapsed >= distance / STRAFE_SPEED

        if done or elapsed >= STRAFE_TIMEOUT:
            if elapsed >= STRAFE_TIMEOUT and not done:
                if self._odom_xy is None:
                    self.get_logger().warn(
                        "Strafe timed out – no odom received on "
                        f"{ODOM_TOPIC}; check the topic name"
                    )
                else:
                    moved = (math.dist(self._odom_xy, self._strafe_start)
                             if self._strafe_start is not None else 0.0)
                    self.get_logger().warn(
                        f"Strafe timed out – odom moved only {moved*1000:.0f} mm "
                        f"of {distance*1000:.0f} mm in {elapsed:.1f} s. Base not "
                        f"responding to {BASE_CMD_TOPIC}, or odom ignores lateral "
                        f"motion. (start={self._strafe_start}, now={self._odom_xy})"
                    )
            self._stop_base()
            if self._strafe_timer:
                self._strafe_timer.cancel()
                self._strafe_timer = None
            self._draw_next_letter()
            return

        twist = Twist()
        twist.linear.y = STRAFE_SIGN * STRAFE_SPEED
        self.base_pub.publish(twist)

    def _stop_base(self) -> None:
        self.base_pub.publish(Twist())

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- Forward / inverse kinematics ---------------------------------------
    # Derived and numerically verified against the real Mirte Master URDF.
    # Coordinates are in the robot BASE frame: +x = forward (robot front),
    # +y = left, z = height (frame_link). Joint variables are sent to the
    # controller directly (no extra negation).
    #   pan   = shoulder_pan_joint   lift = shoulder_lift_joint
    #   elbow = elbow_joint          wrist = wrist_joint
    # lift/elbow are measured from vertical (0 = link points up).
    #
    # The shoulder pan is limited to ±90° and its zero points to the BACK of
    # the robot, so reaching the FRONT requires the mirrored arm fold:
    # negative reach, with negative lift and elbow. Drawing therefore happens
    # in a band roughly 0.265–0.335 m in front of the robot.

    def _fk_base(self, pan: float, lift: float, elbow: float) -> tuple[float, float, float]:
        """Forward kinematics to the wrist origin → base (forward, left, z)."""
        s2, c2   = math.sin(lift), math.cos(lift)
        s23, c23 = math.sin(lift + elbow), math.cos(lift + elbow)
        reach = L1 * s2 + L2 * s23 - REACH_OFFSET
        forward = SHOULDER_Y - math.cos(pan) * reach
        left    = -math.sin(pan) * reach
        z = SHOULDER_Z + L1 * c2 + L2 * c23
        return forward, left, z

    def _ik(self, forward: float, lateral: float, z: float) -> tuple | None:
        """
        Inverse kinematics for the wrist origin, reaching toward the robot
        FRONT. (forward, lateral, z) are base-frame coords (+forward, +left).
        Returns (pan, lift, elbow, wrist) within ±90° limits, or None if the
        target is unreachable / would violate a joint limit.
        """
        cx = forward - SHOULDER_Y
        reach = -math.hypot(cx, lateral)   # negative → front (mirrored) fold
        pan = math.atan2(lateral, cx)

        a = reach + REACH_OFFSET          # horizontal component of the 2-link
        b = z - SHOULDER_Z                # vertical component (from pivot)

        cos_elbow = (a * a + b * b - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if abs(cos_elbow) > 1.0:
            return None                   # out of physical reach

        # Try both elbow configurations; keep the first within all joint limits.
        for sign in (-1.0, 1.0):
            elbow = math.atan2(sign * math.sqrt(max(0.0, 1.0 - cos_elbow ** 2)), cos_elbow)
            lift  = math.atan2(a, b) - math.atan2(
                L2 * math.sin(elbow),
                L1 + L2 * math.cos(elbow),
            )
            if (abs(pan)   <= JOINT_LIMIT and
                    abs(lift)  <= JOINT_LIMIT and
                    abs(elbow) <= JOINT_LIMIT):
                return pan, lift, elbow, WRIST_DRAW_ANGLE
        return None

    def _selftest(self) -> None:
        """Log a round-trip IK/FK check and the reachable forward band."""
        center = (DRAW_X_CENTER, 0.0, WRIST_DRAW_HEIGHT)
        sol = self._ik(*center)
        if sol is None:
            self.get_logger().error(
                f"IK self-test FAILED: draw center {center} is UNREACHABLE. "
                f"Move DRAW_X_CENTER into the reachable band below."
            )
        else:
            pan, lift, elbow, wrist = sol
            fwd, lat, z = self._fk_base(pan, lift, elbow)
            err = math.dist((fwd, lat, z), center)
            self.get_logger().info(
                f"IK self-test OK: center {center} → "
                f"pan={math.degrees(pan):.1f}° lift={math.degrees(lift):.1f}° "
                f"elbow={math.degrees(elbow):.1f}°  (round-trip err {err*1000:.2f} mm)"
            )

        lo = hi = None
        f = 0.05
        while f < 0.45:
            if self._ik(f, 0.0, WRIST_DRAW_HEIGHT) is not None:
                lo = f if lo is None else lo
                hi = f
            f += 0.005
        if lo is not None:
            self.get_logger().info(
                f"Reachable forward band at wrist z={WRIST_DRAW_HEIGHT}: "
                f"{lo:.3f}–{hi:.3f} m (lateral=0). Letters span "
                f"{DRAW_X_CENTER - LETTER_HEIGHT/2:.3f}–{DRAW_X_CENTER + LETTER_HEIGHT/2:.3f} m."
            )

    # -- Helpers -------------------------------------------------------------

    def _send_arm(self, pan: float, lift: float, elbow: float,
                  wrist: float, duration: float) -> None:
        traj = JointTrajectory()
        traj.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint",        "wrist_joint",
        ]
        pt = JointTrajectoryPoint()
        pt.positions = [pan, lift, elbow, wrist]
        pt.time_from_start.sec = max(1, int(duration))
        traj.points.append(pt)
        self.arm_pub.publish(traj)

    def _signal_done(self) -> None:
        msg = String()
        msg.data = "DONE"
        self.state_pub.publish(msg)
        self._mode = "IDLE"
        self.get_logger().info("Drawing complete – signalling DONE")

    def _cancel_timers(self) -> None:
        if self._probe_timer:
            self._probe_timer.cancel()
        if self._done_timer:
            self._done_timer.cancel()
        if self._strafe_timer:
            self._strafe_timer.cancel()
            self._strafe_timer = None
        self._stop_base()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SandDrawer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
