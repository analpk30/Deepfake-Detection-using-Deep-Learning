import numpy as np
import cv2
from app import preprocess_frames, extract_frames


def test_preprocess_frames_and_extract(tmp_path):
    # create a tiny video to test extract_frames
    h, w = 64, 64
    video_path = tmp_path / "v.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
    for _ in range(6):
        frame = (np.random.rand(h, w, 3) * 255).astype("uint8")
        out.write(frame)
    out.release()

    frames = extract_frames(str(video_path), max_frames=4)
    assert isinstance(frames, list)
    assert len(frames) <= 4

    if len(frames) > 0:
        X = preprocess_frames(frames, target_size=(32, 32))
        assert X.dtype == "float32"
        assert X.ndim == 4
        assert X.shape[1:] == (32, 32, 3)
