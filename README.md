# GestureOS

> Control your operating system with nothing but your hand.

GestureOS is a local desktop prototype that maps hand gestures captured by your webcam to real operating system actions — opening a browser, minimising windows, zooming in, navigating browser history — with zero cloud APIs, zero custom model training, and zero latency to an external server.

Everything runs on your machine using MediaPipe's pre-trained Hand Landmarker model and simple geometric heuristics on the 21 detected hand landmarks.

---

## Demo

```
┌──────────────────────────────────────────────┐
│ Gesture: OPEN PALM                           │
│ ████████████████████████████████████░░░░░░░ │  ← confidence bar
│ Action: Opened browser → https://google.com  │
│                                              │
│   [live webcam feed with hand skeleton]      │
│                                              │
│                              Press Q to quit │
└──────────────────────────────────────────────┘
```

---

## Gesture Reference

| Gesture | How to perform | OS Action | Hotkey sent |
|---|---|---|---|
| **Open Palm** | Hold all 4 fingers straight up, thumb out | Opens Google in default browser | `webbrowser.open()` |
| **Fist** | Curl all fingers into a closed fist | Minimises current window | `Win + ↓` |
| **Pinch** | Touch thumb tip to index tip, other fingers extended | Zoom in | `Ctrl + +` |
| **Swipe Left** | Move hand quickly to the left | Browser back | `Alt + ←` |
| **Swipe Right** | Move hand quickly to the right | Browser forward | `Alt + →` |

### Stability & cooldown

- A gesture must be held **consistently for 5 frames** before it fires. This prevents false positives from transition frames between gestures.
- A **2-second cooldown** follows every triggered action. The same gesture cannot re-fire until the cooldown elapses, so you can hold your hand still without spamming.
- An orange progress bar in the overlay shows how many confirmation frames have been collected (fills left-to-right).

---

## How it works

### Hand landmark detection

MediaPipe's Hand Landmarker model returns 21 3-D keypoints per hand in every frame:

```
                8   12  16  20
                |   |   |   |
            7   |   |   |   19
        6   |  11  15  18   |
    5   |  10   |  14   |  17
    |   9   |  13   |  16   |
    |   |   |   |   |   |   |
    ----+---+---+---+---+---+--- (wrist = 0)
```

All coordinates are normalised to `[0, 1]` within the image, so detection is resolution-independent.

### Gesture logic (no ML classifier needed)

Each gesture is detected by pure geometry on those 21 points:

| Gesture | Rule |
|---|---|
| **Open palm** | Tips of all 4 fingers (`y`) are above their PIP joints |
| **Fist** | Tips of all 4 fingers are below their MCP knuckles |
| **Pinch** | Euclidean distance between thumb tip and index tip, normalised by palm width, < 0.35; middle finger partially extended (to distinguish from a fist) |
| **Swipe** | Horizontal displacement of index tip > 18% of frame width over the last 0.5 seconds |

### Pipeline per frame

```
Webcam → flip (mirror) → RGB convert → MediaPipe Hand Landmarker
    → 21 landmarks → gesture classifier + swipe tracker
    → stability buffer (5 frames) → cooldown check → OS action
    → overlay render → imshow
```

---

## Project structure

```
GestureOS/
├── main.py               # All application logic (~220 lines)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore
└── hand_landmarker.task  # Downloaded at setup (not tracked in git)
```

### `main.py` module breakdown

| Section | Lines | Responsibility |
|---|---|---|
| Config block | top | Tunable constants — thresholds, cooldown, URL |
| Landmark indices | constants | Named aliases for the 21 MediaPipe keypoints |
| `is_open_palm / is_fist / is_pinch` | functions | Pure boolean gesture classifiers |
| `detect_gesture` | function | Dispatch: calls classifiers in priority order |
| `SwipeDetector` | class | Rolling time-window swipe tracker |
| `execute_action` | function | Maps gesture string → OS hotkey / webbrowser call |
| `draw_landmarks` | function | Draws hand skeleton on frame with OpenCV |
| `draw_overlay` | function | Draws HUD (gesture label, progress bar, action label) |
| `main` | function | Camera loop, MediaPipe session, stability + cooldown logic |

---

## Requirements

- **Python 3.10 – 3.14** (tested on 3.14.3 with MediaPipe 0.10.35)
- A working webcam (built-in or USB)
- Windows 10 / 11 (hotkeys use `Win + ↓` and `Alt + ←/→`; see [Platform notes](#platform-notes) for Mac/Linux)
- Internet connection for the one-time model download (~9 MB)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/amitkarmakar1980/GestureOS.git
cd GestureOS
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe model

The hand landmark model file is ~9 MB and is not tracked in git. Download it once:

```bash
python -c "
import urllib.request
url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
urllib.request.urlretrieve(url, 'hand_landmarker.task')
print('Model downloaded.')
"
```

---

## Running

```bash
python main.py
```

A window titled **GestureOS** opens with the live webcam feed.

| Overlay element | Meaning |
|---|---|
| Green text — top left | Currently detected gesture name |
| Orange progress bar | Confirmation progress (fills over 5 stable frames) |
| Yellow text | Last triggered OS action |
| Grey text — bottom right | Reminder to press Q |

Press **Q** inside the GestureOS window to quit cleanly.

---

## Tuning

All tuneable constants are at the top of `main.py`:

| Constant | Default | Effect |
|---|---|---|
| `GESTURE_CONFIRM_FRAMES` | `5` | Higher = less false positives, more latency |
| `COOLDOWN_SECONDS` | `2.0` | Minimum gap between repeated action fires |
| `BROWSER_URL` | `https://google.com` | URL opened by the open-palm gesture |

To change the swipe sensitivity, edit these inside `SwipeDetector`:

| Constant | Default | Effect |
|---|---|---|
| `HISTORY_SECONDS` | `0.5` | How long a swipe window is |
| `SWIPE_THRESHOLD` | `0.18` | Minimum displacement (18% of frame width) |

---

## Platform notes

| Feature | Windows | macOS | Linux |
|---|---|---|---|
| Minimise window | `Win + ↓` ✅ | Replace with `Cmd + M` | Use `xdotool` |
| Zoom in | `Ctrl + +` ✅ | Replace with `Cmd + +` | `Ctrl + +` ✅ |
| Browser back | `Alt + ←` ✅ | Replace with `Cmd + [` | `Alt + ←` ✅ |
| Browser forward | `Alt + →` ✅ | Replace with `Cmd + ]` | `Alt + →` ✅ |

**macOS adaptation** — in `execute_action()` replace:
```python
pyautogui.hotkey("win", "down")   →  pyautogui.hotkey("command", "m")
pyautogui.hotkey("alt", "left")   →  pyautogui.hotkey("command", "[")
pyautogui.hotkey("alt", "right")  →  pyautogui.hotkey("command", "]")
```

**Linux** — PyAutoGUI requires `python3-xlib` or `python3-tk`:
```bash
sudo apt install python3-xlib
```

---

## Dependency overview

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.9 | Webcam capture, frame rendering, display window |
| `mediapipe` | ≥ 0.10.14 | Hand landmark detection (21 keypoints per hand) |
| `pyautogui` | ≥ 0.9.54 | Keyboard hotkey simulation, cross-platform |

No GPU required. MediaPipe runs on CPU via TensorFlow Lite XNNPACK.

---

## Known limitations

1. **Single hand only.** The detector is configured for `num_hands=1`. A second hand in frame is ignored.
2. **Swipe sensitivity.** Very fast or very slow swipes can miss. Tune `SWIPE_THRESHOLD` if needed.
3. **Pinch/fist overlap.** In poor lighting the pinch check may fire instead of fist. Move to a well-lit area.
4. **`Win + ↓` targets the focused window.** Make sure the window you want to minimise is in focus before gesturing.
5. **Camera index 0 assumed.** If your webcam is not on index 0, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` (or whichever index your camera uses).
6. **No two-hand support.** Combos requiring both hands are not implemented.
7. **Mediapipe telemetry warning.** A `portable_clearcut_uploader` error may appear in the terminal. This is MediaPipe trying to send anonymous usage data and failing — it is harmless.

---

## Roadmap / next improvements

- [ ] **Close tab gesture** — L-shape / gun hand → `Ctrl + W`
- [ ] **Volume control** — vertical pinch drag up/down → `VK_VOLUME_UP / DOWN`
- [ ] **Two-hand combos** — enable `num_hands=2` and define combination gestures
- [ ] **Left vs. right hand** — use handedness label to differentiate gesture sets
- [ ] **Trained classifier** — replace geometry heuristics with a small scikit-learn or TFLite model for more robust recognition
- [ ] **Config file** — YAML/JSON mapping of gesture → action so non-developers can customise without editing code
- [ ] **System-tray icon** — pause/resume gesture detection from the taskbar
- [ ] **macOS & Linux hotkey abstraction** — single config layer instead of manual hotkey edits
- [ ] **Gesture recording** — capture landmark sequences to build a training dataset for custom gestures

---

## License

MIT — see [LICENSE](LICENSE).
