# ISL Gesture Detector

This is a real-time Indian Sign Language (ISL) alphabet detector built using Python, OpenCV, and MediaPipe. It supports static hand gestures using one or two hands and maps them to ISL letters like A, B, C, etc.

## ✨ Features

- Detects ISL static signs from hand gestures
- Works with both one-hand and two-hand inputs
- Real-time video feed and letter overlay
- Built with MediaPipe for accurate tracking

## 🧠 How It Works

- Tracks hand landmarks using MediaPipe.
- Calculates which fingers are extended.
- Matches the finger states with known ISL patterns.
- Displays the detected alphabet on screen.

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
