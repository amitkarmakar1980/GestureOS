"""
GestureOS - Hand gesture to OS action mapper
Uses MediaPipe Hands + OpenCV to detect gestures and trigger system actions.
"""

import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
from collections import deque

# ── Configuration ─────────────────────────────────────────────────────────────

GESTURE_CONFIRM_FRAMES = 5   # frames gesture must be stable before triggering
COOLDOWN_SECONDS = 2.0       # seconds between consecutive action fires
BROWSER_URL = "https://google.com"

# ── MediaPipe setup ───────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── Gesture detection helpers ─────────────────────────────────────────────────

# MediaPipe landmark indices
WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
INDEX_MCP   = 5
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17
THUMB_IP    = 3
INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18


def finger_extended(lm, tip, pip):
    """Return True if a finger tip is above its PIP joint (finger is extended)."""
    return lm[tip].y < lm[pip].y


def count_extended_fingers(lm):
    """Count how many of the 4 non-thumb fingers are extended."""
    fingers = [
        finger_extended(lm, INDEX_TIP,  INDEX_PIP),
        finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        finger_extended(lm, RING_TIP,   RING_PIP),
        finger_extended(lm, PINKY_TIP,  PINKY_PIP),
    ]
    return sum(fingers)


def is_open_palm(lm):
    """All 4 fingers extended AND thumb roughly extended to the side."""
    return count_extended_fingers(lm) == 4


def is_fist(lm):
    """All 4 fingers curled (tips below their MCP knuckles)."""
    curled = [
        lm[INDEX_TIP].y  > lm[INDEX_MCP].y,
        lm[MIDDLE_TIP].y > lm[MIDDLE_MCP].y,
        lm[RING_TIP].y   > lm[RING_MCP].y,
        lm[PINKY_TIP].y  > lm[PINKY_MCP].y,
    ]
    return all(curled)


def is_pinch(lm):
    """
    Thumb tip and index tip are close together while the other fingers
    are not fully curled (distinguishes from a fist).
    Distance is normalised by the palm width (wrist-to-index-MCP).
    """
    palm_size = abs(lm[WRIST].x - lm[INDEX_MCP].x) + abs(lm[WRIST].y - lm[INDEX_MCP].y)
    if palm_size < 1e-6:
        return False
    dx = lm[THUMB_TIP].x - lm[INDEX_TIP].x
    dy = lm[THUMB_TIP].y - lm[INDEX_TIP].y
    dist = (dx**2 + dy**2) ** 0.5
    normalised = dist / palm_size
    # Middle+ fingers should be at least partially extended so it isn't a fist
    middle_up = lm[MIDDLE_TIP].y < lm[MIDDLE_MCP].y
    return normalised < 0.35 and middle_up


def detect_gesture(lm):
    """
    Evaluate landmarks and return a gesture string, or None if unrecognised.
    Order matters: pinch before fist because a pinch has curled fingers too.
    """
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
    Tracks the horizontal position of the index-finger tip across recent frames
    and fires 'swipe_left' or 'swipe_right' when the displacement crosses a
    threshold within a short time window.
    """

    HISTORY_SECONDS = 0.5   # look-back window
    SWIPE_THRESHOLD = 0.18  # normalised x displacement (0–1 across frame width)

    def __init__(self):
        # Each entry: (timestamp, x_normalised)
        self._history: deque = deque(maxlen=30)

    def update(self, lm) -> str | None:
        now = time.time()
        x = lm[INDEX_TIP].x          # MediaPipe gives 0–1 across frame width
        self._history.append((now, x))

        # Only keep recent entries
        cutoff = now - self.HISTORY_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 4:
            return None

        oldest_x = self._history[0][1]
        newest_x = self._history[-1][1]
        delta = newest_x - oldest_x   # positive = moved right in frame

        # Note: MediaPipe x increases left→right, but the camera image is
        # mirrored for display, so a right-hand swipe to the right in real
        # life appears as decreasing x in the raw feed.
        if delta < -self.SWIPE_THRESHOLD:
            self._history.clear()
            return "swipe_right"   # hand moved right (real-world)
        if delta > self.SWIPE_THRESHOLD:
            self._history.clear()
            return "swipe_left"    # hand moved left  (real-world)
        return None


# ── OS action dispatcher ──────────────────────────────────────────────────────

def execute_action(gesture: str) -> str:
    """Map a confirmed gesture to an OS action and return a display label."""
    pyautogui.FAILSAFE = False   # prevent corner-of-screen abort during demo

    if gesture == "open_palm":
        webbrowser.open(BROWSER_URL)
        return f"Opened browser → {BROWSER_URL}"

    if gesture == "fist":
        pyautogui.hotkey("win", "down")   # Windows: minimize
        return "Minimised window"

    if gesture == "pinch":
        # Zoom in: Ctrl + Plus (works in browsers, file explorers, etc.)
        pyautogui.hotkey("ctrl", "+")
        return "Zoomed in (Ctrl++)"

    if gesture == "swipe_left":
        pyautogui.hotkey("alt", "left")   # browser back
        return "Browser ← Back"

    if gesture == "swipe_right":
        pyautogui.hotkey("alt", "right")  # browser forward
        return "Browser → Forward"

    return ""


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlay(frame, gesture: str, action: str, confirming: int):
    h, w = frame.shape[:2]
    # Semi-transparent banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    gesture_label = gesture.replace("_", " ").upper() if gesture else "—"
    cv2.putText(frame, f"Gesture: {gesture_label}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Confirmation progress bar
    if confirming > 0:
        bar_w = int((confirming / GESTURE_CONFIRM_FRAMES) * (w - 20))
        cv2.rectangle(frame, (10, 35), (10 + bar_w, 50), (0, 200, 255), -1)

    if action:
        cv2.putText(frame, f"Action: {action}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    cv2.putText(frame, "Press Q to quit",
                (w - 170, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0). Check camera permissions.")

    swipe_detector = SwipeDetector()

    # Stability buffer: counts how many consecutive frames the same gesture appeared
    gesture_buffer: str | None = None
    gesture_count: int = 0

    last_action_time: float = 0.0
    last_action_label: str = ""

    print("GestureOS running — press Q in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame so it feels like a mirror (left/right as expected)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected: str | None = None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            lm = hand_lm.landmark  # list of 21 NormalizedLandmark objects

            # Check swipe first (motion-based, overrides static gestures)
            swipe = swipe_detector.update(lm)
            if swipe:
                detected = swipe
            else:
                detected = detect_gesture(lm)

        # ── Stability / confirmation logic ────────────────────────────────────
        if detected == gesture_buffer and detected is not None:
            gesture_count += 1
        else:
            gesture_buffer = detected
            gesture_count = 1 if detected else 0

        confirmed = gesture_count >= GESTURE_CONFIRM_FRAMES

        # ── Cooldown + fire ───────────────────────────────────────────────────
        now = time.time()
        if confirmed and (now - last_action_time) >= COOLDOWN_SECONDS:
            last_action_label = execute_action(gesture_buffer)
            last_action_time = now
            gesture_count = 0   # reset so the same gesture must re-stabilise

        # ── Overlay ───────────────────────────────────────────────────────────
        draw_overlay(frame, gesture_buffer, last_action_label,
                     min(gesture_count, GESTURE_CONFIRM_FRAMES))

        cv2.imshow("GestureOS", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("GestureOS stopped.")


if __name__ == "__main__":
    main()
