import time
import os
import json

import cv2
import numpy as np
import pytesseract
from PIL import Image

IMAGE_PATH = os.path.join("images", "text_frame.jpg")
DEBUG_PATH = os.path.join("images", "whiteboard_crop.jpg")

# PSM 7: single text line; uppercase + pipe (| is read as I for hand-drawn text)
TESSERACT_CONFIG = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ|"


def _order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _perspective_warp(gray, quad):
    rect = _order_points(quad)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (w, h))


def _crop_whiteboard(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray

    best_quad = None
    best_area = 0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < 0.05 * gray.size:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx
            best_area = area

    if best_quad is not None:
        return _perspective_warp(gray, best_quad)

    # fallback: bounding rect of the largest bright region
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return gray[y:y + h, x:x + w]

    return gray


def preprocess(image_path: str) -> Image.Image:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    warped = _crop_whiteboard(gray)

    # Find where the whiteboard surface actually starts/ends using column brightness.
    # Columns that are mostly dark are background (shelving, people, tripod).
    h, w = warped.shape
    col_mean = warped.mean(axis=0)        # mean brightness per column
    bright = col_mean > 140               # True where column is mostly bright (whiteboard)
    left_cols = np.where(bright)[0]
    x_left = int(left_cols[0]) if left_cols.size else int(w * 0.15)
    x_right = int(left_cols[-1]) if left_cols.size else int(w * 0.75)
    # Add inward margin so we don't clip onto the dark boundary pixels
    margin = int(w * 0.03)
    x_left += margin
    x_right -= margin
    # Clamp to reasonable bounds so we don't eat into the text
    x_left = min(x_left, int(w * 0.25))
    x_right = max(x_right, int(w * 0.60))
    cropped = warped[0:int(h * 0.75), x_left:x_right]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped = clahe.apply(cropped)
    cropped = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 41, 15)

    cv2.imwrite(DEBUG_PATH, cropped)

    # Scale to 200px height so letters are ~80-120px — optimal for Tesseract
    ch, cw = cropped.shape
    cropped = cv2.resize(cropped, (int(cw * 200 / ch), 200), interpolation=cv2.INTER_AREA)
    cropped = cv2.copyMakeBorder(cropped, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    return Image.fromarray(cropped)


def run_ocr(image_path: str) -> dict:
    img = preprocess(image_path)
    w_img, _ = img.size

    data = pytesseract.image_to_data(img, config=TESSERACT_CONFIG,
                                     output_type=pytesseract.Output.DICT)

    tokens = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word or int(data["conf"][i]) < 0:
            continue
        cx = data["left"][i] + data["width"][i] / 2
        # Reject tokens whose center is in the outer 20% — those are border noise
        if cx < w_img * 0.20 or cx > w_img * 0.80:
            continue
        # Normalize pipe/l to I (hand-drawn I looks like | to Tesseract)
        tokens.append((data["left"][i], word.replace("|", "I").replace("l", "I")))

    tokens.sort(key=lambda t: t[0])
    text = "".join(w for _, w in tokens)
    return {"text": text, "words": []}


def main():
    last_mtime: float | None = None

    while True:
        if not os.path.exists(IMAGE_PATH):
            time.sleep(0.05)
            continue

        mtime = os.path.getmtime(IMAGE_PATH)
        if mtime == last_mtime:
            time.sleep(0.05)
            continue

        last_mtime = mtime

        try:
            output = run_ocr(IMAGE_PATH)
            print(json.dumps(output), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()