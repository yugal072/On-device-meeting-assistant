import time
import json
import mss
import cv2
import numpy as np
import easyocr


INTERVAL = 5      # seconds between screenshots
DURATION = 120    # total capture time (2 min for testing)

reader = easyocr.Reader(['en'], gpu=False)
sct = mss.mss()

monitor = sct.monitors[1]

results = []
start_time = time.time()


def capture_screen():
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


print("Starting screen + OCR capture...")


while time.time() - start_time < DURATION:

    t = round(time.time() - start_time, 2)

    frame = capture_screen()

    # Optional: save frames for debugging
    cv2.imwrite(f"frame_{int(t)}.png", frame)

    ocr_result = reader.readtext(frame)

    texts = []

    for box, text, conf in ocr_result:
        if conf > 0.4:
            texts.append(text)

    joined_text = " ".join(texts)

    results.append({
        "time": t,
        "ocr": joined_text
    })

    print(f"[{t}s] Text found: {len(texts)}")

    time.sleep(INTERVAL)


with open("vision.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Done. Saved vision.json")
