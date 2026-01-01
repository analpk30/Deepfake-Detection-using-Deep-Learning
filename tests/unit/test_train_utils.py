import os
import cv2
import numpy as np
from train_utils import read_frames_from_dir


def test_read_frames_from_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    h, w = 32, 32
    # create 10 fake images
    for i in range(10):
        img = (np.random.rand(h, w, 3) * 255).astype("uint8")
        cv2.imwrite(str(d / f"img_{i:02d}.png"), img)

    frames = read_frames_from_dir(str(d), max_frames=5)
    assert isinstance(frames, list)
    assert len(frames) == 5

    # test empty dir
    empty = tmp_path / "empty"
    empty.mkdir()
    frames_empty = read_frames_from_dir(str(empty), max_frames=4)
    assert frames_empty == []
