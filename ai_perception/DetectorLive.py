"""
DetectorLive.py
---------------
Standalone detector script that runs inside the project's AI venv.

Launched as a subprocess by ``detector_goal_node.py``; communicates via
stdio:
  stdout – one JSON detection dict per line (consumed by the ROS node's
           _stdout_reader thread)
  stderr – warnings and errors (consumed by the ROS node's
           _stderr_reader thread)

Keeping the two streams separate means the ROS node can parse stdout as
pure JSON without needing to filter out error messages.

Detection dict schema (one per detected object, per frame)
──────────────────────────────────────────────────────────
  {
    "class":      str,    object class label
    "confidence": float,  model confidence [0, 1]
    "center_x":   float,  horizontal bbox centre (pixels)
    "center_y":   float,  vertical   bbox centre (pixels)
    "width":      float,  bbox width  (pixels)
    "height":     float,  bbox height (pixels)
  }

Environment variables
─────────────────────
  ROBOFLOW_API_KEY  – required; Roboflow API key for the inference client
"""

import json
import os
import signal
import sys
import time

from inference_sdk import InferenceHTTPClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path where detector_goal_node.py writes the latest camera frame.
# Must match FRAME_PATH in detector_goal_node.py.
IMAGE_PATH: str = os.path.join("images", "frame.jpg")

# Polling interval (s) when no new frame is available
POLL_INTERVAL: float = 0.1

# Roboflow workspace / workflow identifiers
WORKSPACE_NAME: str = "spatial-ai-kdgzb"
WORKFLOW_ID: str = "general-segmentation-api-7"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Thin wrapper around the Roboflow InferenceHTTPClient."""

    def __init__(self, api_key: str) -> None:
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

    def run_inference(self, image_path: str) -> list:
        """
        Run the configured Roboflow workflow on *image_path*.

        Returns the raw workflow result list.  ``use_cache=False`` ensures
        the model always processes the latest frame rather than returning a
        cached response.
        """
        return self.client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": image_path},
            parameters={"classes": "Whiteboard"},
            use_cache=False,
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _emit(obj: dict) -> None:
    """Serialise *obj* to stdout as a single JSON line (flush immediately)."""
    print(json.dumps(obj), flush=True)


def _warn(message: str) -> None:
    """Write *message* to stderr so the ROS node's stderr reader picks it up."""
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Poll for new camera frames, run inference, and emit detections to stdout.

    Shutdown
    ────────
    The loop exits cleanly on SIGTERM (sent by the ROS node when it calls
    ``process.terminate()``) or on KeyboardInterrupt.
    """
    # ── API key ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        _warn(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Export it before launching this script."
        )
        sys.exit(1)

    detector = Detector(api_key)

    # ── Graceful SIGTERM handling ─────────────────────────────────────────────
    # The ROS node terminates this subprocess with SIGTERM; converting it to a
    # KeyboardInterrupt lets the loop exit via the normal except clause below.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    last_mtime: float | None = None

    _warn("DetectorLive ready – waiting for frames")

    # ── Polling loop ──────────────────────────────────────────────────────────
    try:
        while True:
            # Wait until the ROS node has written at least one frame
            if not os.path.exists(IMAGE_PATH):
                time.sleep(POLL_INTERVAL)
                continue

            mtime = os.path.getmtime(IMAGE_PATH)
            if mtime == last_mtime:
                # Frame unchanged since last inference – poll again
                time.sleep(POLL_INTERVAL)
                continue

            last_mtime = mtime

            try:
                result = detector.run_inference(IMAGE_PATH)
                _parse_and_emit(result)

            except Exception as exc:  # noqa: BLE001
                # Log inference errors to stderr; do NOT emit to stdout so the
                # ROS node's JSON parser is not confused by non-detection output.
                _warn(f"Inference error: {exc}")

    except KeyboardInterrupt:
        _warn("DetectorLive shutting down")


def _parse_and_emit(result: list) -> None:
    """
    Extract per-object predictions from a workflow result and emit each one
    as a JSON line on stdout.

    The result structure is:
      result[0]["predictions"]["predictions"] → list of prediction dicts

    Malformed or unexpected structures are logged to stderr and skipped.
    """
    try:
        predictions: list = result[0]["predictions"]["predictions"]
    except (IndexError, KeyError, TypeError) as exc:
        _warn(f"Unexpected result structure: {exc} | raw: {result!r}")
        return

    for obj in predictions:
        try:
            detection = {
                "class":      obj["class"],
                "confidence": float(obj["confidence"]),
                "center_x":   float(obj["x"]),
                "center_y":   float(obj["y"]),
                "width":      float(obj["width"]),
                "height":     float(obj["height"]),
            }
            _emit(detection)
        except (KeyError, ValueError, TypeError) as exc:
            _warn(f"Malformed prediction entry: {exc} | obj: {obj!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()