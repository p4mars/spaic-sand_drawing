#!/usr/bin/env python3

import json
import math

import cv2
import numpy as np
from cv2 import aruco

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ------------------------------------------------------------------
# Camera parameters
# ------------------------------------------------------------------

FOV_DEG = 60.0
FOV_RAD = math.radians(FOV_DEG)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

TARGET_MARKER_ID = 10

ACTIVE_STATE = "TRACK_WHITEBOARD"


# ------------------------------------------------------------------
# Node
# ------------------------------------------------------------------

class ArucoGoalNode(Node):

    def __init__(self):
        super().__init__("aruco_goal_node")

        self.bridge = CvBridge()
        self.current_state: str = ""

        self.target_pub = self.create_publisher(
            String,
            "/target_angle",
            10
        )

        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10
        )

        self.create_subscription(
            String,
            "/robot_state",
            self._state_callback,
            10
        )

        # ArUco setup
        self.dictionary = aruco.getPredefinedDictionary(
            aruco.DICT_4X4_50
        )

        self.detector = aruco.ArucoDetector(
            self.dictionary
        )

        self.get_logger().info(
            f"ArucoGoalNode started. Tracking marker ID {TARGET_MARKER_ID}"
        )

    # --------------------------------------------------------------

    def _state_callback(self, msg: String) -> None:
        self.current_state = msg.data

    # --------------------------------------------------------------

    def image_callback(self, msg):

        if self.current_state != ACTIVE_STATE:
            return

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is None:
                return

            for marker_corners, marker_id in zip(
                corners,
                ids.flatten()
            ):
                if marker_id != TARGET_MARKER_ID:
                    continue
                pts = marker_corners[0]

                # Marker center
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))

                # Approximate size
                width = np.linalg.norm(
                    pts[0] - pts[1]
                )

                height = np.linalg.norm(
                    pts[1] - pts[2]
                )

                bbox_area = float(width * height)

                self.publish_target(
                    cx,
                    cy,
                    bbox_area
                )

                # Only process first matching marker
                break

        except Exception as e:
            self.get_logger().warn(
                f"Detection error: {e}"
            )

    # --------------------------------------------------------------

    def publish_target(
        self,
        cx,
        cy,
        bbox_area
    ):

        error_x = -(cx - IMAGE_WIDTH / 2)
        angle_x = (
            error_x / IMAGE_WIDTH
        ) * FOV_RAD

        error_y = -(cy - IMAGE_HEIGHT / 2)
        angle_y = (
            error_y / IMAGE_HEIGHT
        ) * FOV_RAD

        msg = String()

        msg.data = json.dumps({
            "angle_x": float(angle_x),
            "angle_y": float(angle_y),
            "bbox_area": float(bbox_area)
        })

        self.target_pub.publish(msg)

        self.get_logger().debug(
            f"Marker detected | "
            f"angle_x={math.degrees(angle_x):.2f} deg | "
            f"bbox_area={bbox_area:.0f}"
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = ArucoGoalNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()