import os
import numpy as np
import cv2
from app import predict_video


class DummyModel:
    def predict(self, batch):
        return np.zeros((len(batch), 1))


def test_predict_smoke(monkeypatch, tmp_path):
    # create a tiny synthetic video
    h, w = 64, 64
    video_path = tmp_path / "vid.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
    for _ in range(4):
        frame = (np.random.rand(h, w, 3) * 255).astype("uint8")
        out.write(frame)
    out.release()

    # monkeypatch model loader to avoid heavy TF model load
    monkeypatch.setattr('app.load_trained_model', lambda path='CNN_model.h5': DummyModel())

    res = predict_video(str(video_path), model_path='dummy', max_frames=4, use_face_detection=False)
    assert isinstance(res, dict)
    assert set(res.keys()) >= {"label", "score", "frames_used"}
