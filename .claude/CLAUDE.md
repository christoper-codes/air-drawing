# Air Drawing — Computer Vision Project

## Stack
- Python 3.13.7
- Virtual environment located at `/venv`
- OpenCV (`opencv-python`) for camera capture and rendering
- MediaPipe for hand landmark tracking
- NumPy for geometric calculations

## Architecture
- `main.py` → application entry point, camera loop, gesture processing
- `hand_tracker.py` → reusable `HandTracker` abstraction for landmark detection
- Standalone desktop application (no Django, Flask, or web frameworks)

## Code Standards
- All code must be written in English
- Use descriptive names for variables, functions, classes, and files
- Keep functions focused and small
- One gesture = one dedicated function
- Prefer composition over duplication
- Refactor aggressively when logic becomes repetitive
- Avoid unnecessary abstractions

## Comments
- Comments must be written in English
- Avoid redundant or obvious comments
- Use comments only when explaining non-trivial logic, assumptions, or constraints
- Prefer self-documenting code over comments

### Activate virtual environment
```bash
venv\Scripts\activate
```

### Run application
```bash
python main.py
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Current Status
- [x] Basic hand detection
- [x] Index finger drawing gesture
- [x] Open hand erase gesture (clears full canvas)
- [x] Two-finger color switching (6 colors, cycles with cooldown)
```