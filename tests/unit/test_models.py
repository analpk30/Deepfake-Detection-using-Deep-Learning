import tensorflow as tf
from models import build_xception


def test_build_xception_returns_model():
    # Use weights=None to avoid downloading pretrained weights in CI
    model = build_xception(input_shape=(128, 128, 3), pretrained=None, freeze_base=True)
    assert isinstance(model, tf.keras.Model)
