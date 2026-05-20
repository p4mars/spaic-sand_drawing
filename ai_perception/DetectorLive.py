# DetectorLive.py  (runs inside the venv)
import time
import os
import json

from inference_sdk import InferenceHTTPClient

IMAGE_PATH = os.path.join("images", "frame.jpg")


class Detector:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="XYZ"  # TODO: replace with your actual API key,
        )

    def run_inference(self, image_path: str):
        return self.client.run_workflow(
            workspace_name="spatial-ai-kdgzb",
            workflow_id="general-segmentation-api-3",
            images={"image": image_path},
            parameters={"classes": "Whiteboard"},
            use_cache=False,   # always want a fresh result
        )


def main():
    detector = Detector()
    last_mtime: float | None = None

    while True:
        # Wait until the ROS node has written at least one frame
        if not os.path.exists(IMAGE_PATH):
            time.sleep(0.05)
            continue

        mtime = os.path.getmtime(IMAGE_PATH)
        if mtime == last_mtime:
            # No new frame yet — busy-wait lightly
            time.sleep(0.05)
            continue

        last_mtime = mtime

        try:
            result = detector.run_inference(IMAGE_PATH)
            predictions = result[0]["predictions"]["predictions"]

            for obj in predictions:
                msg = {
                    "class": obj["class"],
                    "confidence": float(obj["confidence"]),
                    "center_x": float(obj["x"]),
                    "center_y": float(obj["y"]),
                    "width": float(obj["width"]),
                    "height": float(obj["height"]),
                }
                # json.dumps produces valid JSON; flush ensures the pipe
                # is not buffered so the ROS node sees it immediately
                print(json.dumps(msg), flush=True)

        except Exception as e:
            # Surface errors to the ROS logger via the same pipe
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()