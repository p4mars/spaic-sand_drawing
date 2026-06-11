import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import subprocess
import json
import threading
import cv2
import os
import sys

from pathlib import Path

workspace = Path.cwd()
base_dir = workspace.parent.parent


class TextDetectionNode(Node):

    def __init__(self):
        super().__init__("text_detection_node")

        self.pub = self.create_publisher(String, "/whiteboard_text", 10)
        self.state_pub = self.create_publisher(String, "/state_change", 10)

        self.bridge = CvBridge()
        os.makedirs("images", exist_ok=True)
        self.image_path = os.path.join("images", "text_frame.jpg")
        self.tmp_path = self.image_path + ".tmp"
        self.ocr_debug_path = os.path.join("images", "ocr_input.jpg")

        self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        self.create_subscription(String, "/robot_state", self.state_callback, 10)

        self.process = None
        self.active = False

    def state_callback(self, msg):
        if msg.data == "READ_SANDPIT" and not self.active:
            self._start_detector()
        elif msg.data != "READ_SANDPIT" and self.active:
            self._stop_detector()

    def image_callback(self, msg: Image):
        if not self.active:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=-30)
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                self.get_logger().warn("imencode failed")
                return
            with open(self.tmp_path, "wb") as f:
                f.write(buffer.tobytes())
            os.replace(self.tmp_path, self.image_path)
            cv2.imwrite(self.ocr_debug_path, frame)
        except Exception as e:
            self.get_logger().warn(f"Image save error: {e}")

    def _read_loop(self):
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                self._process_result(result)
            except Exception as e:
                self.get_logger().warn(f"Parse error: {e} | raw: {line}")

    def _process_result(self, result):
        if "error" in result:
            self.get_logger().warn(f"Detector error: {result['error']}")
            return

        text = result.get("text", "")
        if not text:
            return

        msg = String()
        msg.data = json.dumps(result)
        self.pub.publish(msg)
        self.get_logger().info(f"Detected text: {text!r}")

        done = String()
        done.data = "DONE"
        self.state_pub.publish(done)
        self._stop_detector()

    def _start_detector(self):
        ai_env_python = base_dir / "ai_env" / "bin" / "python"
        python_path = ai_env_python if ai_env_python.exists() else Path(sys.executable)
        detector_path = base_dir / "ai_perception" / "TextDetectorLive.py"

        self.process = subprocess.Popen(
            [str(python_path), str(detector_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.active = True
        self.get_logger().info(f"Started text detector PID: {self.process.pid}")
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _stop_detector(self):
        if self.process:
            self.get_logger().info(f"Stopping text detector PID: {self.process.pid}")
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.active = False

    def destroy_node(self):
        self._stop_detector()
        super().destroy_node()


def main():
    rclpy.init()
    node = TextDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()