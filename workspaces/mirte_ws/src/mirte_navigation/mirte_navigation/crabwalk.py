import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import time


class CrabWalk(Node):

    def __init__(self, direction="left"):
        super().__init__('crab_walk_node')

        self.publisher_ = self.create_publisher(
            Twist,
            '/mirte_base_controller/cmd_vel_unstamped',
            10
        )

        self.distance = 0.1  # 10 cm
        self.speed = 0.05

        self.direction = direction

    def execute(self):
        twist = Twist()

        if self.direction == "left":
            twist.linear.y = self.speed
        elif self.direction == "right":
            twist.linear.y = -self.speed
        elif self.direction == "forward":
            twist.linear.x = self.speed
        elif self.direction == "backward":
            twist.linear.x = -self.speed

        duration = self.distance / self.speed
        start = time.time()

        while rclpy.ok() and time.time() - start < duration:
            self.publisher_.publish(twist)
            time.sleep(0.05)

        # stop robot
        self.publisher_.publish(Twist())
        self.get_logger().info("Finished crab walk")


def main(args=None):
    rclpy.init(args=args)

    direction = "left"
    if len(sys.argv) > 1:
        direction = sys.argv[1]

    node = CrabWalk(direction)
    node.execute()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()