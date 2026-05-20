import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class StateManager(Node):

    def __init__(self):
        super().__init__('state_manager')
        # ── Publishers ──────────────────────────────────────────────────────
        self.state_pub = self.create_publisher(
            String,
            '/robot_state',
            10
        )

        self.state_change_pub = self.create_publisher(
            String,
            '/state_change',
            10
        )

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/mirte_master_arm_controller/joint_trajectory",
            10,
        )
        # ── Subscribers ─────────────────────────────────────────────────────
        self.done_sub = self.create_subscription(
            String,
            '/state_change',
            self.state_callback,
            10
        )
        # ── Main control loop ────────────────────────────────────────
        self.current_state = "RAISE_ARM"
        msg = String()
        msg.data = self.current_state
        self.state_change_pub.publish(msg)
        self.timer = self.create_timer(0.5, self.state_publisher_callback)
        # Small delay before starting actions
        self.timer = self.create_timer(
            7.0,
            self.start_sequence
        )


    def raise_arm(self) -> None:
        """
        Send a single joint-trajectory command to fold the arm upward so that
        it does not occlude the forward LiDAR beam.
        """

        traj = JointTrajectory()
        traj.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_joint",
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            0.0,    # shoulder_pan  – centred
            0.0,    # shoulder_lift – down
            -1.56,  # elbow         – folded back (~90 °) to clear LiDAR
            0.0,    # wrist         – neutral
        ]
        point.time_from_start.sec = 7  # allow 7 s for the motion to complete

        traj.points.append(point)
        self.arm_pub.publish(traj)
        self.get_logger().info("Arm raised for LiDAR clearance")    
        self.current_state = "TRACK_WHITEBOARD"
        msg = String()
        msg.data = self.current_state
        self.state_change_pub.publish(msg)
        self.get_logger().info(
            f"State -> {self.current_state}"
        )      

    def state_callback(self,msg):
        self.current_state = msg.data
        print(self.current_state)
        if self.current_state == "TRACK_WHITEBOARD" :
            self.get_logger().info('Tracking Whiteboard')

        if msg.data == "DONE":
            msg = String()
            msg.data = self.current_state
            self.state_pub.publish(msg)
            self.get_logger().info("Shutting down")
            rclpy.shutdown()
            return
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)
        
    def state_publisher_callback(self):
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    def start_sequence(self):
        # Run once
        self.timer.cancel()

        self.raise_arm()


def main(args=None):

    rclpy.init(args=args)

    node = StateManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()