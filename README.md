# ISL Gesture Detector

Real-time Indian Sign Language (ISL) alphabet detector using Python, OpenCV, and MediaPipe. Supports static hand gestures with one or two hands.

## Features

- Detects ISL static signs from hand gestures
- Works with one-hand and two-hand inputs
- Real-time video feed with letter and FPS overlay
- Built with MediaPipe for accurate landmark tracking

## How It Works

- Tracks hand landmarks using MediaPipe
- Calculates which fingers are extended
- Matches finger states against known ISL patterns
- Displays detected alphabet on screen

## Supported Letters

| One Hand | Two Hand |
|----------|----------|
| A, B, C, D, E, I, L, U, V, W, Y | M, W, J, BOTH B |

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

## Usage

```bash
python detector.py
```

Press **Q** to quit. Press **S** to save a screenshot.

## Project Structure

```
├── detector.py      # Main detection logic
├── config.py        # Configuration constants
├── utils.py         # Drawing utilities
├── main.py          # Entry point
├── requirements.txt # Dependencies
└── README.md        # Documentation
```
