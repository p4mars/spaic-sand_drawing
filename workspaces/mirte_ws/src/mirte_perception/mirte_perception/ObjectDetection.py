# detector_goal_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import subprocess
import json
import threading
import math
import cv2
import os


from pathlib import Path

# Start from the current workspace location
workspace = Path.cwd()

# Go up from mirte_ws → workspaces → spatial_ai
base_dir = workspace.parent.parent

# adjust the number depending on where this file lives

class DetectorGoalNode(Node):

    def __init__(self):
        super().__init__("detector_goal_node")

        self.pub = self.create_publisher(String, "/target_angle", 10)

        # Camera parameters
        self.image_width = 640
        self.fov_deg = 60.0
        self.fov_rad = math.radians(self.fov_deg)

        self.bridge = CvBridge()

        # Write to a temp file, then atomically rename so the
        # detector never reads a half-written image
        self.image_path = os.path.join("images", "frame.jpg")
        self.tmp_path = self.image_path + ".tmp"

        # Subscribe to camera — adjust topic name to match your setup
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.state_sub = self.create_subscription(
            String,
            '/robot_state',
            self.state_callback,
            10
        )
        self.initialized = False

    # ------------------------------------------------------------------

    def state_callback(self,msg):
        if msg.data == "TRACK_WHITEBOARD" and not self.initialized:
            self.initialize_model()
            self.initialized = True
        if msg.data != "TRACK_WHITEBOARD" and self.initialized:
             self.initialized = False
             self.get_logger().info(f"Ending detector PID: {self.process.pid}")
             self.process.terminate()
             self.process.wait()


    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # imencode avoids the extension-lookup bug in minimal OpenCV builds
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                self.get_logger().warn("imencode failed")
                return

            with open(self.tmp_path, "wb") as f:
                f.write(buffer.tobytes())

            os.replace(self.tmp_path, self.image_path)

        except Exception as e:
            self.get_logger().warn(f"Image save error: {e}")

    # ------------------------------------------------------------------

    def read_loop(self):
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                self.get_logger().info(f"No Line")
                continue
            try:
                detection = json.loads(line)           # detector now sends valid JSON
                self.process_detection(detection)
            except Exception as e:
                self.get_logger().warn(f"Parse error: {e} | raw: {line}")

    def process_detection(self, detection):
        if detection.get("class") != "Whiteboard":
            self.get_logger().info(f"No Whiteboard")
            return

        cx = detection["center_x"]
        bbox_area = detection["width"] / detection["height"]
        error_px = -(cx - (self.image_width / 2))
        angle = (error_px / self.image_width) * self.fov_rad

        msg = String()
        msg.data = json.dumps({
            "angle": angle,
            "bbox_area": bbox_area
        })
        self.pub.publish(msg)

        self.get_logger().info(f"Published angle: {math.degrees(angle):.2f}°")

    # ------------------------------------------------------------------

    def initialize_model (self):
        python_path = base_dir / "ai_env" / "bin" / "python"
        detector_path = base_dir / "ai_perception" / "DetectorLive.py"

        # Start detector subprocess (inside its venv)
        self.process = subprocess.Popen(
            [
                str(python_path),
                str(detector_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.get_logger().info(f"Started detector PID: {self.process.pid}")

        threading.Thread(target=self.read_loop, daemon=True).start()


    def destroy_node(self):
        self.process.terminate()
        self.process.wait()
        super().destroy_node()


def main():
    rclpy.init()
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