import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow")
pytest.importorskip("streamlit")

from app import aggregate_scores, extract_frames, predict_video, preprocess_frames


class DummyModel:
    def predict(self, batch):
        return np.zeros((len(batch), 1))


def test_extract_and_preprocess(tmp_path):
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


def test_aggregate_scores_mean_median_majority():
    scores = [0.1, 0.9, 0.8, 0.2]
    s_mean, l_mean = aggregate_scores(scores, method="mean", threshold=0.5)
    assert abs(s_mean - np.mean(scores)) < 1e-6
    assert l_mean == 1

    s_median, l_median = aggregate_scores(scores, method="median", threshold=0.5)
    assert abs(s_median - np.median(scores)) < 1e-6
    assert l_median == 1

    s_maj, l_maj = aggregate_scores(scores, method="majority", threshold=0.5)
    assert abs(s_maj - 0.5) < 1e-6
    assert l_maj == 1


def test_predict_smoke(monkeypatch, tmp_path):
    h, w = 64, 64
    video_path = tmp_path / "vid.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
    for _ in range(4):
        frame = (np.random.rand(h, w, 3) * 255).astype("uint8")
        out.write(frame)
    out.release()

    monkeypatch.setattr(
        "app.load_trained_model", lambda path="CNN_model.h5": DummyModel()
    )
    res = predict_video(
        str(video_path),
        model_path="dummy",
        max_frames=4,
        use_face_detection=False,
    )
    assert isinstance(res, dict)
    assert set(res.keys()) >= {"label", "score", "frames_used"}
