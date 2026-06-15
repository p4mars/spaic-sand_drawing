# Python Environment Setup

The `ai_env/` virtual environment folder is **excluded from this repository** (it is machine-specific, architecture-dependent, and hundreds of MB). Follow the steps below to recreate it locally.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 (Jammy) |
| Python | 3.10+ |
| ROS 2 | Humble |

Make sure ROS 2 is sourced in your shell before proceeding:

```bash
source /opt/ros/humble/setup.bash
```

---

## 1. Create and activate the virtual environment

```bash
# From the repo root
python3 -m venv ai_env

# Activate (Linux / macOS)
source ai_env/bin/activate

# Activate (Windows — only for local dev, not supported on Mirte)
# ai_env\Scripts\activate
```

---

## 2. Install project dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify the setup

```bash
python3 -c "import cv2, ultralytics, supervision; print('All imports OK')"
```

---

## Notes

- **ROS 2 packages** (`rclpy`, `cv_bridge`, `tf2_ros`, etc.) are installed **system-wide via apt** and are NOT in `requirements.txt`. They do not go inside the venv.
- **Do not commit `ai_env/`** — it is listed in `.gitignore`.
- If you install a new package during development, update `requirements.txt`:
  ```bash
  # Only freeze packages explicitly installed, not the full system
  pip freeze | grep -v "^-e" > requirements.txt
  # Then manually remove any ROS/system packages that snuck in
  git add requirements.txt
  ```
- The `inference-sdk` package (Roboflow) is included but note it has a **large install size** (~500MB+ with dependencies). On the Mirte robot this may take a while.

---

## .gitignore entry

Make sure your `.gitignore` contains:

```
ai_env/
__pycache__/
*.pyc
*.egg-info/
```
