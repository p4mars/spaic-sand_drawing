import cv2
import pytesseract
from PIL import Image

TESSERACT_CONFIG = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def run_ocr(frame):
    processed = preprocess(frame)
    text = pytesseract.image_to_string(Image.fromarray(processed), config=TESSERACT_CONFIG).strip()
    return text


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Live OCR running — press Q to quit")
    last_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        text = run_ocr(frame)

        if text and text != last_text:
            print(f"Detected: {text!r}")
            last_text = text

        display = frame.copy()
        y = 30
        for line in text.splitlines():
            if line.strip():
                cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y += 35

        cv2.imshow("Live OCR", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
