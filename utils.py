"""Utility functions for drawing and display."""
import cv2


def draw_text(img, text, pos, scale=0.6, color=(200, 200, 200), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
