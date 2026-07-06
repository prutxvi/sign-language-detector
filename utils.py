"""Utility functions for the ISL detector."""
"""Utility functions for drawing and display."""
import cv2


def draw_text(img, text, pos, scale=0.6, color=(200, 200, 200), thickness=1, font=None):
    font = font or cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, scale, color, thickness)
