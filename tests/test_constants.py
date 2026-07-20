import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import constants


def test_core_constants_values():
    assert constants.DEFAULT_MODEL_PATH == "CNN_model.h5"
    assert constants.DEFAULT_DATA_ROOT == "Dataset"
    assert constants.DEFAULT_TARGET_SIZE == (128, 128)
    assert constants.LABEL_REAL == "Real"
    assert constants.LABEL_FAKE == "Fake"
