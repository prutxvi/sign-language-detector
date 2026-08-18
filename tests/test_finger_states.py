"""Tests for finger state extraction and ISL mapping."""

from detector import ISL_ONE_HAND, ISL_TWO_HAND, get_finger_states


class Landmark:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


def make_hand(thumb_x, tip_y, base_y):
    """Build 21 landmarks with controllable thumb and finger positions."""
    lm = [Landmark() for _ in range(21)]
    lm[3].x = 0.5
    lm[4].x = thumb_x
    for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        lm[tip].y = tip_y
        lm[base].y = base_y
    return lm


def test_all_fingers_extended():
    lm = make_hand(thumb_x=0.9, tip_y=0.1, base_y=0.9)
    assert get_finger_states(lm, is_right=True) == (1, 1, 1, 1, 1)


def test_fist_no_fingers_extended():
    lm = make_hand(thumb_x=0.4, tip_y=0.9, base_y=0.1)
    assert get_finger_states(lm, is_right=True) == (0, 0, 0, 0, 0)


def test_thumb_only():
    lm = make_hand(thumb_x=0.9, tip_y=0.9, base_y=0.1)
    assert get_finger_states(lm, is_right=True) == (1, 0, 0, 0, 0)


def test_left_hand_thumb_mirrors():
    lm = make_hand(thumb_x=0.1, tip_y=0.9, base_y=0.1)
    assert get_finger_states(lm, is_right=False) == (1, 0, 0, 0, 0)


def test_one_hand_mapping_a():
    assert ISL_ONE_HAND[(1, 0, 0, 0, 0)] == "A"


def test_two_hand_mapping_m():
    assert ISL_TWO_HAND[((1, 0, 0, 0, 0), (1, 0, 0, 0, 0))] == "M"
