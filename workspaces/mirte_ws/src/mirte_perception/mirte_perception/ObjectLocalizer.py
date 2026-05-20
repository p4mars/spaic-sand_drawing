import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
import math


class VisionGoalGenerator(Node):

    def __init__(self):
        super().__init__("vision_goal_generator")

        self.sub = self.create_subscription(
            String,
            "/detections",
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10
        )

        # camera params (adjust later if needed)
        self.image_width = 640
        self.fov_deg = 60.0
        self.fov_rad = math.radians(self.fov_deg)

        self.distance = 3.0  # meters forward goal
        print("VisionGoalGenerator initialized with FOV:", self.fov_deg, "degrees and goal distance:", self.distance, "meters")

    def callback(self, msg):
        detections = json.loads(msg.data)

        # pick best whiteboard detection
        target = None
        if detections["class"] == "Whiteboard":
            target = detections
        if target is None:
            self.stop()
            return
        self.compute_goal(target)


    def compute_goal(self, target):

        cx = target["center_x"]

        # 1. pixel error
        error_px = -(cx - (self.image_width / 2))

        # 2. convert to angle
        angle = (error_px / self.image_width) * self.fov_rad

        # 3. create goal in robot frame
        goal = PoseStamped()

        goal.pose.position.x = self.distance * math.cos(angle)
        goal.pose.position.y = self.distance * math.sin(angle)

        # orientation = yaw toward target
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)

        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.pub.publish(goal)

        self.get_logger().info(
            f"Goal: angle={math.degrees(angle):.2f} deg | x={goal.pose.position.x:.2f}"
        )


def main():
    rclpy.init()
    node = VisionGoalGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()