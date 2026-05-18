# Sand Drawing Robot — AE4ASM527 Spatial AI
**TU Delft · Group 1**

A fully autonomous robot system that reads a pattern from a whiteboard and reproduces it in a sandbox. Built on the Mirte Master Robot 2 with ROS2 Humble.

---

## Table of contents
- [Project overview](#project-overview)
- [System architecture](#system-architecture)
- [Getting started](#getting-started)
- [Running the system](#running-the-system)
- [Development workflow](#development-workflow)

---

## Project overview

The robot autonomously executes the following pipeline:

1. **Navigate** through the room and localize the whiteboard
2. **Detect and interpret** the pattern on the whiteboard (text or geometric shape)
3. **Navigate** to the sandbox
4. **Reproduce** the pattern in the sand using the gripper

**Robot:** Mirte Master Robot 2  
**Framework:** ROS2 Humble  
**Simulation:** Gazebo + RViz  
**Motion planning:** MoveIt2  
**Perception:** OpenCV, ArUco markers  

---

## System architecture

```
Robot hardware (camera · sensors · encoders · gripper)
        │
        ├── /scan ──────────────► slam_node (WB · Module 1)
        │                              │
        ├── /odom ──────────────►      │── /map
        │                              │── /robot_pose
        │                              │
        └── /camera/image_raw ──► whiteboard_detector (SA · Module 3)
                                       │── /whiteboard_pose
                                       │── /pattern_coordinates
                                       │
                          task_coordinator (WT · Module 2)
                          ┌────────────┴────────────┐
                NavigateToPose (action)     MoveGroup (action)
                          │                          │
               navigation_node (WB)    gripper_controller (JC · Module 4)
                    │                              │
               /cmd_vel                  /joint_trajectory
```

| Node | Owner | Module | Description |
|------|-------|--------|-------------|
| `slam_node` | WB | 1 | SLAM via Nav2 — builds map and tracks robot pose |
| `navigation_node` | WB | 1 | Drives robot to goal poses |
| `task_coordinator` | WT | 2 | State machine: navigate → detect → draw |
| `whiteboard_detector` | SA | 3 | Detects whiteboard and recognizes pattern via ArUco / OpenCV |
| `gripper_controller` | JC | 4 | Plans and executes drawing path via MoveIt2 |

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

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build
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
source install/setup.bash
ros2 launch launch/simulation.launch.py
```

### Real robot

```bash
source install/setup.bash
ros2 launch launch/robot.launch.py
```

### Run individual nodes

```bash
# SLAM only
ros2 run slam slam_node

# Whiteboard detection only
ros2 run perception whiteboard_detector

# Task coordinator (full pipeline)
ros2 run task_planning task_coordinator
```

---

## Development workflow

We use Scrum with 1-week sprints managed in Trello.

### Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable, tested code only — never push directly |
| `dev` | Integration branch — merge features here first |
| `feature/slam` | Module 1 — WB |
| `feature/detection` | Module 3 — SA |
| `feature/gripper` | Module 4 — JC |
| `feature/integration` | Module 2 — WT |

### Pull request rules
- Always branch from `dev`, not `main`
- At least 1 reviewer must approve before merging
- Code must be tested in simulation before merging to `dev`
- Code must be tested on the real Mirte robot before merging to `main`
