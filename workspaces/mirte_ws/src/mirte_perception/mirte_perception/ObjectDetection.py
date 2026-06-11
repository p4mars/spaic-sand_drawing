"""
detector_goal_node.py
---------------------
ROS 2 node that bridges a vision-model subprocess and the robot's control
pipeline.

Responsibilities
────────────────
  1. Subscribes to /robot_state; starts the external detector subprocess
     when the state is TRACK_WHITEBOARD and terminates it otherwise.
  2. Saves the latest camera frame to disk (atomic write) so the detector
     subprocess can read it without race conditions.
  3. Parses JSON detections from the subprocess stdout and publishes the
     horizontal angle to the detected whiteboard on /target_angle.

Topics
──────
  Subscribed
    /camera/image_raw  (sensor_msgs/Image)  – raw camera frames
    /robot_state       (std_msgs/String)    – current robot state
  Published
    /target_angle      (std_msgs/String)    – JSON {"angle": float, "bbox_area": float}

Subprocess path convention
──────────────────────────
  Paths are resolved relative to *this file*, not the working directory,
  so the node works regardless of where it is launched from:

    <repo_root>/
      ai_env/                ← Python venv for the detector
      ai_perception/
        DetectorLive.py      ← detector script
      mirte_ws/
        src/
          <package>/
            detector_goal_node.py   ← this file
"""

import json
import math
import os
import subprocess
import threading
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Resolve paths relative to this source file so the node works regardless
# of the working directory at launch time.
#
# Expected layout (adjust _LEVELS_UP if this file moves):
#   <base_dir>/
#     ai_env/bin/python
#     ai_perception/DetectorLive.py
#     mirte_ws/src/<pkg>/detector_goal_node.py   ← this file (3 levels up)

# Start from the current workspace location
workspace = Path.cwd()
# Go up from mirte_ws → workspaces → spatial_ai
BASE_DIR: Path = workspace.parent.parent

PYTHON_PATH: Path = BASE_DIR / "ai_env" / "bin" / "python"
DETECTOR_PATH: Path = BASE_DIR / "ai_perception" / "DetectorLive.py"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Camera horizontal field of view (degrees → radians)
FOV_DEG: float = 60.0
FOV_RAD: float = math.radians(FOV_DEG)

# Expected image width in pixels (used for angle computation)
IMAGE_WIDTH: int = 640
IMAGE_HEIGHT: int = 480  

# Directory where camera frames are staged for the detector subprocess
FRAME_DIR: str = "images"
FRAME_PATH: str = os.path.join(FRAME_DIR, "frame.jpg")
FRAME_TMP_PATH: str = FRAME_PATH + ".tmp"   # atomic write staging path


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DetectorGoalNode(Node):
    """
    Manages the external vision-model subprocess and translates its
    detections into ROS 2 angle messages.
    """

    # -- Lifecycle -----------------------------------------------------------

    def __init__(self) -> None:
        super().__init__("detector_goal_node")

        # ── Publishers ───────────────────────────────────────────────────────
        self.target_pub = self.create_publisher(String, "/target_angle", 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._image_callback,
            10,
        )

        self.create_subscription(
            String,
            "/robot_state",
            self._state_callback,
            10,
        )

        # ── Internal state ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self._process: subprocess.Popen | None = None   # detector subprocess
        self._detector_active: bool = False             # True while subprocess is running

        # Ensure the frame staging directory exists before any image arrives
        os.makedirs(FRAME_DIR, exist_ok=True)

        self.get_logger().info(
            f"DetectorGoalNode ready\n"
            f"  detector : {DETECTOR_PATH}\n"
            f"  python   : {PYTHON_PATH}"
        )

    # -- State management ----------------------------------------------------

    def _state_callback(self, msg: String) -> None:
        """
        Start or stop the detector subprocess based on the robot state.

        The subprocess is started on the first TRACK_WHITEBOARD message and
        terminated when the state leaves TRACK_WHITEBOARD.
        """
        state = msg.data

        if state == "TRACK_WHITEBOARD" and not self._detector_active:
            self._start_detector()

        elif state != "TRACK_WHITEBOARD" and self._detector_active:
            self._stop_detector()

    # -- Camera frame writer -------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        """
        Convert an incoming ROS image to JPEG and write it atomically to disk.

        The atomic rename (write to .tmp then os.replace) ensures the detector
        subprocess never reads a half-written file.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=-30)

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                self.get_logger().warn("imencode failed – frame dropped")
                return

            with open(FRAME_TMP_PATH, "wb") as f:
                f.write(buffer.tobytes())

            os.replace(FRAME_TMP_PATH, FRAME_PATH)

        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Image save error: {exc}")

    # -- Subprocess management -----------------------------------------------

    def _start_detector(self) -> None:
        """
        Launch the detector script in a subprocess and start a background
        thread to read its stdout line by line.

        A second daemon thread drains stderr so the pipe buffer never fills
        up and blocks the subprocess.
        """
        if not PYTHON_PATH.exists():
            self.get_logger().error(f"Python venv not found: {PYTHON_PATH}")
            return
        if not DETECTOR_PATH.exists():
            self.get_logger().error(f"Detector script not found: {DETECTOR_PATH}")
            return

        self._process = subprocess.Popen(
            [str(PYTHON_PATH), str(DETECTOR_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._detector_active = True
        self.get_logger().info(f"Detector started (PID {self._process.pid})")

        # Daemon threads die automatically when the main process exits
        threading.Thread(target=self._stdout_reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()

    def _stop_detector(self) -> None:
        """Terminate the detector subprocess and reset the active flag."""
        if self._process is None:
            return

        self.get_logger().info(f"Stopping detector (PID {self._process.pid})")
        self._process.terminate()
        self._process.wait()
        self._process = None
        self._detector_active = False

    # -- Subprocess I/O readers ----------------------------------------------

    def _stdout_reader(self) -> None:
        """
        Background thread: read JSON detections from the subprocess stdout.

        Runs until the subprocess closes its stdout (i.e. exits).
        Empty lines are silently skipped; malformed lines are logged as
        warnings so they don't crash the thread.
        """
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue  # empty line – skip silently

            try:
                detection = json.loads(line)
                self._process_detection(detection)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"stdout parse error: {exc} | raw: {line!r}")

    def _stderr_reader(self) -> None:
        """
        Background thread: drain the subprocess stderr to prevent pipe
        buffer overflow, and forward any output as ROS warnings.
        """
        for line in self._process.stderr:
            line = line.strip()
            if line:
                self.get_logger().warn(f"[detector stderr] {line}")

    # -- Detection processing ------------------------------------------------

    def _process_detection(self, detection: dict) -> None:
        """
        Convert a raw detector dict into a /target_angle message.

        Expected detection fields
        ─────────────────────────
          class    (str)   – object class label
          center_x (float) – horizontal pixel position of the bounding-box centre
          width    (float) – bounding-box width  in pixels
          height   (float) – bounding-box height in pixels

        Angle convention
        ────────────────
          Positive angle → target is to the LEFT  of the image centre.
          Negative angle → target is to the RIGHT of the image centre.
          (The negation below converts from image-x convention, where right
           is positive, to robot convention, where left is positive.)
        """
        if detection.get("class") != "Whiteboard":
            return  # ignore other detected classes

        cx: float = detection["center_x"]
        cy: float = detection["center_y"]
        bbox_area: float = detection["width"] * detection["height"]  # pixels²

        # Pixel error (positive = target is left of centre)
        error_px: float = -(cx - IMAGE_WIDTH / 2)
        angle_x: float = (error_px / IMAGE_WIDTH) * FOV_RAD
        error_px: float = -(cy - IMAGE_HEIGHT / 2)
        angle_y: float = (error_px / IMAGE_HEIGHT) * FOV_RAD

        msg = String()
        msg.data = json.dumps({"angle_x": angle_x, "angle_y": angle_y, "bbox_area": bbox_area})
        self.target_pub.publish(msg)

        self.get_logger().debug(
            f"Whiteboard detected – angle {math.degrees(angle_x):+.2f}°  "
            f"bbox_area {bbox_area:.0f} px²"
        )

    # -- Node teardown -------------------------------------------------------

    def destroy_node(self) -> None:
        """Ensure the detector subprocess is terminated before the node exits."""
        self._stop_detector()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """Initialise ROS 2, spin the node, and clean up on exit."""
    rclpy.init(args=args)
    node = DetectorGoalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()