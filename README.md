# GestureOS

A local desktop prototype that maps hand gestures captured by your webcam to operating system actions.
No cloud APIs, no custom ML model — just MediaPipe's pre-trained hand landmark detector plus simple geometry heuristics.

---

## Gesture → Action map

| Gesture | How to perform it | OS action |
|---|---|---|
| **Open palm** | Hold all 4 fingers straight up | Opens `https://google.com` in default browser |
| **Fist** | Curl all fingers into a fist | Minimises the current window (`Win + ↓`) |
| **Pinch** | Touch thumb tip to index tip, other fingers out | Zoom in (`Ctrl ++`) |
| **Swipe left** | Move hand quickly to the left | Browser back (`Alt + ←`) |
| **Swipe right** | Move hand quickly to the right | Browser forward (`Alt + →`) |

A gesture must be held consistently for **5 frames** before it fires, and there is a **2-second cooldown** between actions.

---

## Requirements

- Python 3.10 or 3.11 (MediaPipe does not yet support 3.13 on Windows)
- A working webcam
- Windows 10/11 (hotkeys use `Win + ↓` and `Alt + ←/→`; see platform notes below)

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> **Tip — Python version:** run `python --version` first. If you have 3.12+,
> install Python 3.11 from python.org and create the venv with it:
> `py -3.11 -m venv .venv`

---

## Running

```bash
python main.py
```

A window titled **GestureOS** opens showing the live webcam feed with hand landmarks drawn on it.

- Green text at the top shows the currently detected gesture.
- An orange progress bar fills as the gesture stabilises over 5 frames.
- Yellow text shows the last triggered action.
- Press **Q** inside the window to quit.

---

## Platform notes

| Feature | Windows | macOS | Linux |
|---|---|---|---|
| Minimise window | `Win + ↓` ✅ | Change to `Cmd + M` | Use `xdotool` |
| Zoom in | `Ctrl + +` ✅ | `Cmd + +` | `Ctrl + +` |
| Browser back/forward | `Alt + ←/→` ✅ | `Cmd + [/]` | `Alt + ←/→` |

To adapt for macOS, replace `pyautogui.hotkey("win", "down")` with `pyautogui.hotkey("command", "m")` etc.

---

## Known limitations

1. **Single hand only.** MediaPipe is configured for `max_num_hands=1`.
2. **Swipe sensitivity is fragile.** Very fast or very slow swipes may not register. Tune `SWIPE_THRESHOLD` in `main.py`.
3. **Pinch vs. fist overlap.** In poor lighting the pinch check can mis-fire. Move to a well-lit area.
4. **`Win + ↓` minimises whatever window has focus** — make sure the target window is focused before gesturing.
5. **No gesture for close-tab yet.** `Ctrl + W` can be added trivially.
6. **Camera index 0 assumed.** If your webcam is on a different index, change `cv2.VideoCapture(0)`.
7. **MediaPipe 0.10.x requires Python ≤ 3.12** on Windows at time of writing.

---

## Next improvements after MVP

- [ ] Add `close_tab` gesture (`Ctrl + W`) — e.g. L-shape / gun hand
- [ ] Detect left vs. right hand and allow two-hand combos
- [ ] Add volume control via vertical pinch drag
- [ ] Replace geometry heuristics with a small trained classifier (scikit-learn or TFLite)
- [ ] Config file (YAML/JSON) so gesture→action mappings are editable without touching code
- [ ] System-tray icon + toggle to pause gesture detection
- [ ] On-screen gesture gallery / tutorial overlay
- [ ] macOS + Linux hotkey abstraction layer
