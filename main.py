"""
GestureOS - Hand gesture to OS action mapper
Uses MediaPipe Hand Landmarker (Tasks API) + OpenCV to detect gestures
and trigger system actions.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import pyautogui
import webbrowser
import time
from collections import deque
import os

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH          = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
GESTURE_CONFIRM_FRAMES = 5    # frames gesture must be stable before triggering
COOLDOWN_SECONDS    = 2.0     # seconds between consecutive action fires
BROWSER_URL         = "https://google.com"

# ── Landmark indices (MediaPipe 21-point hand model) ──────────────────────────

WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8;  INDEX_PIP  = 6;  INDEX_MCP  = 5
MIDDLE_TIP  = 12; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP    = 16; RING_PIP   = 14; RING_MCP   = 13
PINKY_TIP   = 20; PINKY_PIP  = 18; PINKY_MCP  = 17

# Connections for drawing (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]

# ── Gesture detection helpers ─────────────────────────────────────────────────

def finger_extended(lm, tip, pip):
    """Finger tip above its PIP joint → finger is extended (y increases downward)."""
    return lm[tip].y < lm[pip].y


def count_extended_fingers(lm):
    return sum([
        finger_extended(lm, INDEX_TIP,  INDEX_PIP),
        finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        finger_extended(lm, RING_TIP,   RING_PIP),
        finger_extended(lm, PINKY_TIP,  PINKY_PIP),
    ])


def is_open_palm(lm):
    """All 4 fingers extended."""
    return count_extended_fingers(lm) == 4


def is_fist(lm):
    """All 4 finger tips below their MCP knuckles."""
    return all([
        lm[INDEX_TIP].y  > lm[INDEX_MCP].y,
        lm[MIDDLE_TIP].y > lm[MIDDLE_MCP].y,
        lm[RING_TIP].y   > lm[RING_MCP].y,
        lm[PINKY_TIP].y  > lm[PINKY_MCP].y,
    ])


def is_pinch(lm):
    """
    Thumb tip and index tip close together, middle finger at least partially up
    (distinguishes pinch from fist). Distance normalised by palm width.
    """
    palm_size = (abs(lm[WRIST].x - lm[INDEX_MCP].x) +
                 abs(lm[WRIST].y - lm[INDEX_MCP].y))
    if palm_size < 1e-6:
        return False
    dx = lm[THUMB_TIP].x - lm[INDEX_TIP].x
    dy = lm[THUMB_TIP].y - lm[INDEX_TIP].y
    dist = (dx**2 + dy**2) ** 0.5
    middle_up = lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
    return (dist / palm_size) < 0.35 and middle_up


def detect_gesture(lm):
    """Return gesture string or None. Pinch checked before fist (subset of curled fingers)."""
    if is_pinch(lm):
        return "pinch"
    if is_open_palm(lm):
        return "open_palm"
    if is_fist(lm):
        return "fist"
    return None


# ── Swipe detection ───────────────────────────────────────────────────────────

class SwipeDetector:
    """
    Tracks index-finger tip x position over a short window.
    A displacement > SWIPE_THRESHOLD (normalised 0–1) fires a swipe.
    """
    HISTORY_SECONDS = 0.5
    SWIPE_THRESHOLD = 0.18

    def __init__(self):
        self._history: deque = deque(maxlen=30)

    def update(self, lm) -> str | None:
        now = time.time()
        self._history.append((now, lm[INDEX_TIP].x))

        cutoff = now - self.HISTORY_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 4:
            return None

        delta = self._history[-1][1] - self._history[0][1]
        # MediaPipe x: 0=left, 1=right. Camera is mirrored for display,
        # so real-world rightward swipe → decreasing x in raw feed.
        if delta < -self.SWIPE_THRESHOLD:
            self._history.clear()
            return "swipe_right"
        if delta > self.SWIPE_THRESHOLD:
            self._history.clear()
            return "swipe_left"
        return None


# ── OS action dispatcher ──────────────────────────────────────────────────────

def execute_action(gesture: str) -> str:
    pyautogui.FAILSAFE = False

    if gesture == "open_palm":
        webbrowser.open(BROWSER_URL)
        return f"Opened browser → {BROWSER_URL}"
    if gesture == "fist":
        pyautogui.hotkey("win", "down")
        return "Minimised window"
    if gesture == "pinch":
        pyautogui.hotkey("ctrl", "+")
        return "Zoomed in (Ctrl++)"
    if gesture == "swipe_left":
        pyautogui.hotkey("alt", "left")
        return "Browser ← Back"
    if gesture == "swipe_right":
        pyautogui.hotkey("alt", "right")
        return "Browser → Forward"
    return ""


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_landmarks(frame, lm_list):
    """Draw hand skeleton on frame. lm_list: list of NormalizedLandmark."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, (0, 150, 80), 1)


def draw_overlay(frame, gesture: str, action: str, confirm_frames: int):
    h, w = frame.shape[:2]
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 75), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.5, frame, 0.5, 0, frame)

    label = gesture.replace("_", " ").upper() if gesture else "—"
    cv2.putText(frame, f"Gesture: {label}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)

    # Confirmation progress bar
    filled = int((min(confirm_frames, GESTURE_CONFIRM_FRAMES) / GESTURE_CONFIRM_FRAMES) * (w - 20))
    cv2.rectangle(frame, (10, 38), (w - 10, 52), (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (10, 38), (10 + filled, 52), (0, 200, 255), -1)

    if action:
        cv2.putText(frame, f"Action: {action}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    cv2.putText(frame, "Press Q to quit",
                (w - 175, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run: python -c \"import urllib.request; "
            "urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', "
            "'hand_landmarker.task')\""
        )

    # Build HandLandmarker in VIDEO mode (processes frames with timestamps)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0). Check camera permissions.")

    swipe_detector = SwipeDetector()

    gesture_buffer: str | None = None
    gesture_count: int = 0
    last_action_time: float = 0.0
    last_action_label: str = ""

    print("GestureOS running — press Q in the window to quit.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural
        frame_idx += 1

        # MediaPipe expects RGB; timestamp in milliseconds
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        detected: str | None = None

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]   # list of 21 NormalizedLandmark
            draw_landmarks(frame, lm)

            swipe = swipe_detector.update(lm)
            detected = swipe if swipe else detect_gesture(lm)

        # Stability: gesture must appear consecutively for GESTURE_CONFIRM_FRAMES
        if detected == gesture_buffer and detected is not None:
            gesture_count += 1
        else:
            gesture_buffer = detected
            gesture_count = 1 if detected else 0

        # Fire action when stable + cooldown elapsed
        now = time.time()
        if (gesture_count >= GESTURE_CONFIRM_FRAMES and
                (now - last_action_time) >= COOLDOWN_SECONDS):
            last_action_label = execute_action(gesture_buffer)
            last_action_time = now
            gesture_count = 0   # must re-stabilise before firing again

        draw_overlay(frame, gesture_buffer, last_action_label, gesture_count)

        cv2.imshow("GestureOS", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("GestureOS stopped.")


if __name__ == "__main__":
    main()
