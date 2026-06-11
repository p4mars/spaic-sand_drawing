"""
state_manager.py
----------------
ROS 2 node that owns and broadcasts the robot's top-level state.

Responsibilities
────────────────
  1. Raises the arm once at startup (allowing time for the arm controller
     to initialise) and then transitions to TRACK_WHITEBOARD.
  2. Publishes the current state on /robot_state at 2 Hz so all other
     nodes can always read the latest value.
  3. Listens on /state_change for transition requests from other nodes
     (e.g. VisionController signalling DONE).

Topics
──────
  Published
    /robot_state   (std_msgs/String) – current state, broadcast at 2 Hz
    /state_change  (std_msgs/String) – transition requests (this node also
                                       emits them to trigger itself)
  Subscribed
    /state_change  (std_msgs/String) – receives state transitions

State sequence
──────────────
  RAISE_ARM  →  (arm reaches position)  →  TRACK_WHITEBOARD
            →  (whiteboard reached)      →  READ_WHITEBOARD
            →  (reading complete)        →  DONE
"""

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Time (s) given for the arm-raise trajectory to complete before the node
# transitions to TRACK_WHITEBOARD.  Must match (or exceed) the trajectory's
# time_from_start.sec value.
ARM_RAISE_DURATION: int = 5

# Delay (s) after node startup before the arm-raise sequence begins, giving
# the arm controller time to come online.
STARTUP_DELAY: float = 3.0

# Valid state transitions: maps each state to the state a DONE signal advances
# it to, or None if DONE means shut down.
# _NEXT_STATE: dict[str, str | None] = {
#     "RAISE_ARM":        "TRACK_WHITEBOARD",
#     "TRACK_WHITEBOARD": "READ_WHITEBOARD",
#     "READ_WHITEBOARD":  None,   # DONE here → shutdown
# }
_NEXT_STATE: dict[str, str | None] = {
    "RAISE_ARM":        "TRACK_WHITEBOARD",
    "TRACK_WHITEBOARD": "READ_SANDPIT",
    "READ_SANDPIT": "TRACK_SANDPIT",
    "TRACK_SANDPIT": "DRAW_PATTERN",
    "DRAW_PATTERN":  'DONE',   # DONE here → shutdown
}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class StateManager(Node):
    """
    Centralised state publisher for the Mirte robot task pipeline.

    Other nodes read /robot_state to know what the robot should be doing.
    State transitions are driven either internally (arm-raise completion) or
    by external nodes publishing on /state_change.
    """

    # -- Lifecycle -----------------------------------------------------------

    def __init__(self) -> None:
        super().__init__("state_manager")

        # ── Publishers ───────────────────────────────────────────────────────
        # Broadcasts the current state at a fixed rate (heartbeat)
        self.state_pub = self.create_publisher(String, "/robot_state", 10)

        # Used to inject state transitions (both by this node and others)
        self.state_change_pub = self.create_publisher(String, "/state_change", 10)

        # Arm joint-trajectory controller
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/mirte_master_arm_controller/joint_trajectory",
            10,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            String,
            "/state_change",
            self._state_change_callback,
            10,
        )

        # ── Internal state ───────────────────────────────────────────────────
        self.current_state: str = "RAISE_ARM"

        # Announce initial state so other nodes can react immediately
        self._publish_state_change(self.current_state)

        # ── Timers ───────────────────────────────────────────────────────────
        # Heartbeat: re-publish current state at 2 Hz
        self.create_timer(0.5, self._state_heartbeat_callback)

        # One-shot: begin the arm-raise sequence after the startup delay.
        # We keep a reference so we can cancel it once it fires.
        self._start_timer = self.create_timer(STARTUP_DELAY, self._start_sequence)

    # -- State helpers --------------------------------------------------------

    def _publish_state_change(self, state: str) -> None:
        """Publish *state* on /state_change (triggers this node's own callback)."""
        msg = String()
        msg.data = state
        self.state_change_pub.publish(msg)

    def _publish_current_state(self) -> None:
        """Publish the current state on /robot_state."""
        msg = String()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    # -- Arm control ----------------------------------------------------------

    def _raise_arm(self) -> None:
        """
        Send a joint-trajectory command to fold the arm upward so that it
        does not occlude the forward LiDAR beam.

        The transition to TRACK_WHITEBOARD is scheduled via a one-shot timer
        (``ARM_RAISE_DURATION`` seconds) instead of blocking with
        ``time.sleep()``, which would freeze the ROS 2 executor.
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
            -1.57,  # elbow         – folded back (~90 °) to clear LiDAR
            0.0,    # wrist         – neutral
        ]
        point.time_from_start.sec = ARM_RAISE_DURATION

        traj.points.append(point)
        self.arm_pub.publish(traj)
        self.get_logger().info("Arm raise command sent")

        # Schedule the state transition once the arm has had time to move.
        # Using a timer avoids blocking the executor with time.sleep().
        self._arm_done_timer = self.create_timer(
            float(ARM_RAISE_DURATION),
            self._on_arm_raise_complete,
        )

    def _on_arm_raise_complete(self) -> None:
        """
        Timer callback fired after the arm-raise duration has elapsed.
        Cancels itself (one-shot behaviour) and triggers the next state.
        """
        self._arm_done_timer.cancel()
        self.current_state = "TRACK_WHITEBOARD"
        msg = String()
        msg.data = self.current_state
        self.state_change_pub.publish(msg)
        self.get_logger().info(f"State -> {self.current_state}")

    # -- Callbacks ------------------------------------------------------------

    def _state_change_callback(self, msg: String) -> None:
        """
        Handle an incoming state-transition request.

        Plain state names (e.g. "TRACK_WHITEBOARD") set the state directly.
        The special token "DONE" advances the current state using the
        ``_NEXT_STATE`` table; if the table maps to None the task is complete
        and the node exits cleanly.
        """
        new_state = msg.data

        if new_state == "DONE":
            next_state = _NEXT_STATE.get(self.current_state)
            if next_state is None:
                # Final DONE – task complete
                self.get_logger().info(
                    f"DONE received in {self.current_state} – task complete, shutting down"
                )
                self._publish_current_state()
                raise SystemExit
            else:
                self.current_state = next_state
                self.get_logger().info(f"DONE received – State → {self.current_state}")
        else:
            self.current_state = new_state
            self.get_logger().info(f"State → {self.current_state}")

        # Broadcast updated state to all subscribers
        self._publish_current_state()

    def _state_heartbeat_callback(self) -> None:
        """Re-broadcast the current state at 2 Hz so late-joining nodes can catch it."""
        self._publish_current_state()

    # -- Startup sequence -----------------------------------------------------

    def _start_sequence(self) -> None:
        """
        One-shot timer callback that begins the arm-raise sequence.

        Cancels itself immediately so the action only runs once.
        """
        self._start_timer.cancel()
        self.get_logger().info("Starting arm-raise sequence")
        self._raise_arm()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    """Initialise ROS 2, spin the node, and clean up on exit."""
    rclpy.init(args=args)
    node = StateManager()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        # SystemExit is raised by _state_change_callback on final DONE
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()