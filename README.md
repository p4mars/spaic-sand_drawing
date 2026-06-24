# Sand Drawing Robot — AE4ASM527 Spatial AI
**TU Delft · Group 1**

A fully autonomous robot system that reads a pattern from a whiteboard and reproduces it in a sandbox. Built on the Mirte Master Robot 2 with ROS2 Humble.

---

## Table of contents
- [Project overview](#project-overview)
- [System architecture](#system-architecture)
- [Team & contributions](#team--contributions)
- [Getting started](#getting-started)
- [Running the system](#running-the-system)
- [Branches](#branches)

---

## Project overview

The robot autonomously executes the following pipeline:

1. **Navigate** through the room and localize the whiteboard
2. **Detect and interpret** the pattern on the whiteboard (text or geometric shapes)
3. **Navigate** to the sandbox
4. **Reproduce** the pattern in the sand using the robotic arm and pen

**Robot:** Mirte Master Robot 2 (mecanum base + 4-DOF arm)  
**Framework:** ROS2 Humble  
**Simulation:** Gazebo + RViz  
**Motion planning:** MoveIt2  
**Perception:** OpenCV, ArUco markers, Roboflow ML API  

---

## System architecture

The nodes communicate over ROS2 topics, coordinated by a central state machine.

```
Robot hardware  (camera · LiDAR · encoders · arm · gripper)
        │
        ├── /scan ──────────────► WhiteBoardTracker  (Wout)
        │                              │
        ├── /odom ──────────────►      │── /cmd_vel  ──► mecanum base
        │                              │
        └── /camera/image_raw ──► ArucoDetection     (Sebas)
                                  TextDetection
                                       │
                                       │── /target_angle
                                       │── /whiteboard_text
                                       │
                              StateMachine            (Wout)
                              /robot_state
                              ┌────────────┴────────────┐
                              │                          │
                    WhiteBoardTracker            SandDrawer           (Wessel)
                    (drives base)                (arm + pen)
                              │                          │
                         /cmd_vel            /joint_trajectory
```

### State machine flow

```
RAISE_ARM ──► TRACK_WHITEBOARD ──► READ_SANDPIT ──► TRACK_SANDPIT ──► DRAW_PATTERN ──► DONE
```

---

## Team & contributions

| Name | Contribution |
|------|-------------|
| **Wout Barrez** | Robot navigation and driving — whiteboard/sandpit tracking, mecanum base control, state machine orchestration |
| **Wessel Toutenhoofd** | Gripper and pen — robotic arm control, inverse kinematics, sand probing, drawing trajectory execution |
| **Sebas Atzori** | Image and text detection,  OCR text recognition, Roboflow object detection integration (but chose tesseract for final demo). I created the add-webcam-ocr branch with the code from the oral exam |

---

## Getting started

### Prerequisites
- Ubuntu 22.04
- ROS2 Humble
- Gazebo
- MoveIt2
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/Wtoutenhoofd/Spatial-AI-group-1.git
cd Spatial-AI-group-1

# Install ROS2 dependencies
rosdep install --from-paths workspaces/mirte_ws/src --ignore-src -r -y

# Build the workspace
cd workspaces/mirte_ws
colcon build
source install/setup.bash
```

### Connect to the real Mirte robot

```bash
ssh mirte@192.168.42.1
# Password: mirte_mirte
```

---

## Running the system

### Simulation (Gazebo + RViz)

```bash
source workspaces/mirte_ws/install/setup.bash
ros2 launch launch/simulation.launch.py
```

### Real robot

```bash
source workspaces/mirte_ws/install/setup.bash
ros2 launch launch/robot.launch.py
```

### Run individual nodes

```bash
# State machine (full pipeline coordinator)
ros2 run mirte_statemachine StateMachine

# Navigation — whiteboard tracker
ros2 run mirte_navigation WhiteBoardTracker

# Perception — ArUco detection
ros2 run mirte_perception ArucoDetection

# Perception — text detection
ros2 run mirte_perception TextDetection

# Drawing — sand drawer
ros2 run mirte_drawing SandDrawer
```

### AI perception (standalone)

The Roboflow-based models run as external Python processes:

```bash
# Object detection
python3 ai_perception/DetectorLive.py

# Text / OCR detection
python3 ai_perception/TextDetectorLive.py
```

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Stable, tested code |
| `FinalVersion` | Final version of the project |
| `dev` | Integration branch |
| `feature/slam` | Navigation and base control |
| `feature/slam_2` | Navigation rework — unfinished |
| `feature/gripper` | Arm control and sand drawing |
| `feature/text-detection` | OCR and image detection |
| `feature/integration` | State machine and full pipeline |
| `feature/integration-test` | Integration testing |
